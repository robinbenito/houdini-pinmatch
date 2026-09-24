"""Camera Pin Matcher - viewer state (HDA ViewerStateModule).

Locks the viewport to the target camera (through the asset's view proxy camera, which mirrors
the target camera exactly and carries the 2D pan/zoom in its own screen window), draws the plate,
the reference mesh and the pins with drawables, and turns pin edits into live camera solves.
Math, storage and camera IO live in the asset's PythonModule (node.hdaModule()).
"""
import json
import os
import pickle
import time
import types

import numpy as np

import hdefereval
import hou
import viewerstate.utils as su

LABEL = "Camera Pin Matcher"
PICK_PX = 12.0          # pin grab radius
SNAP_PX = 20.0          # point / edge snapping radius when creating pins
ZOOM_RANGE = (0.25, 64.0)
COLORS = {"active": (0.15, 1.0, 0.35), "inactive": (0.6, 0.6, 0.6), "locked": (1.0, 0.55, 0.1),
          "outlier": (1.0, 0.2, 0.2), "selected": (1.0, 0.95, 0.15), "ghost": (0.8, 0.4, 1.0)}
DISPLAYS = (("wire", "Wireframe (All Edges)"), ("hidden", "Hidden Line"), ("ghost", "Hidden Line Ghost"),
            ("shaded", "Shaded"))
PULL = 1e-3             # wires / ghost faces are drawn this much (relative) nearer the eye than the depth faces
GHOST_ALPHA = 0.35
NEAR_FRACTION = 1e-3    # tool view near clip = this x (eye distance to the reference + its radius)
R = hou.uiEventReason

HUD = {
    "title": LABEL, "desc": "tool", "icon": "OBJ_camera",
    "rows": [
        {"id": "pins", "label": "Active pins"},
        {"id": "frame", "label": "Frame"},
        {"id": "status", "label": "Solver"},
        {"id": "rms", "label": "RMS error"},
        {"id": "outliers", "label": "Outlier pins"},
        {"id": "locks", "label": "Locked"},
        {"id": "lens", "label": "Focal / H-FOV"},
        {"id": "keys", "label": "Keyed frames"},
        {"id": "ref", "label": "Reference"},
        {"type": "divider"},
        {"id": "k_create", "label": "Create pin on mesh", "key": "Ctrl + LMB"},
        {"label": "Select / drag pin", "key": "LMB"},
        # "actions": hotkey actions; onEnter shows the keys assigned to them (Hotkey Editor), not the defaults
        {"label": "Delete selected pin", "actions": ("delete_pin",)},
        {"label": "Toggle active / locked", "actions": ("toggle_active", "toggle_lock")},
        {"label": "Solve & key", "actions": ("solve_key",)},
        {"label": "Copy pins from nearest keyed frame", "actions": ("copy_nearest",)},
        {"label": "2D pan / zoom", "key": "MMB / mousewheel"},
        {"label": "Reset 2D view", "actions": ("reset_view",)},
        {"label": "Plate mode / mesh display / ghosts", "actions": ("toggle_mode", "cycle_display", "toggle_ghosts")},
    ]}

# Hotkeyable menu actions: id -> (label, default keys). Del / Backspace also reach onKeyEvent first.
ACTIONS = {
    "toggle_active": ("Toggle Pin Active", "A"), "toggle_lock": ("Toggle Pin Lock", "L"),
    "delete_pin": ("Delete Pin", ("Del", "Backspace")), "deactivate_worst": ("Deactivate Worst Pin", ()),
    "solve_key": ("Solve & Key", "K"), "copy_nearest": ("Copy Pins from Nearest Keyed Frame", "C"),
    "reset_view": ("Reset 2D View", "H"), "toggle_mode": ("Toggle Plate Mode", "M"),
    "cycle_display": ("Cycle Mesh Display", "W"), "toggle_ghosts": ("Show Ghost Pins", "G"),
}


def _vec(v):
    return hou.Vector3(float(v[0]), float(v[1]), float(v[2]))


def _pull(eye, a):
    """Scale about the eye by 1 - a: points keep their image position but move a (relative) toward
    the camera, so lines win the depth test against the faces they lie on (drawables have no
    polygon offset)."""
    k = 1.0 - a
    return hou.Matrix4(((k, 0.0, 0.0, 0.0), (0.0, k, 0.0, 0.0), (0.0, 0.0, k, 0.0),
                        (eye[0] * a, eye[1] * a, eye[2] * a, 1.0)))


def _hotkey(state_name, action):
    """The hotkey symbol su.defineHotkey gives a menu action."""
    return "%s.%s" % (su.hotkeyContextForState(state_name, hou.objNodeTypeCategory()), action)


def _later(fn):
    """Run fn when Houdini is idle; quietly skip it if the tool's node was deleted in the meantime."""
    def run():
        try:
            fn()
        except hou.ObjectWasDeleted:
            pass
    hdefereval.executeDeferred(run)


def _count(n):
    return "%.1fM" % (n / 1e6) if n >= 1e6 else "%dk" % round(n / 1e3) if n >= 1e4 else str(n)


class State(object):
    def __init__(self, state_name, scene_viewer):
        self.state_name = state_name
        self.sv = scene_viewer
        self.node = self.pm = self.vp = None
        self.drag = None          # active pin drag
        self.pan = None           # active MMB pan
        self.selected = None      # selected pin id (UI only)
        self.saved = None         # viewport state to restore on exit
        self.hud_values = None
        self.active = False       # between onEnter and onExit; deferred work checks it
        self.watch_pending = False
        self.caches = {}
        self.last_press = (None, 0.0)   # (pixel, time) of the last LMB press, to drop a trailing 'Picked'


    # ---------------------------------------------------------------- lifecycle
    def onEnter(self, kwargs):
        self.active = True
        self.node = kwargs["node"]
        self.pm = self.node.hdaModule()
        kwargs["state_flags"]["indirect_handle_drag"] = False     # MMB is ours (2D pan)
        self.vp = self.sv.curViewport()
        self._make_drawables()
        # The viewport's visible-object mask before the tool hid anything. A session that never got to
        # onExit (asset reinstalled while active, crash) left its exclusions in the viewport; keeping
        # the original on the node stops the next session from taking those for the user's mask.
        mask = self.node.parm("view_mask").eval() or self.vp.settings().visibleObjects()
        ref = self.pm.reference_object(self.node)
        if ref is not None:          # hidden by earlier versions that re-attached after exiting
            mask = " ".join(t for t in mask.split() if t != "^" + ref.path()) or "*"
        with hou.undos.disabler():
            self.node.parm("view_mask").set(mask)
        self.saved = {"camera": self.vp.camera(), "linked": self.vp.isCameraLockedToView(),
                      "view": self.vp.defaultCamera().stash(), "mask": mask}
        self._set_view(0.5, 0.5, 1.0)
        self._attach()
        self._vp_callback = self._on_viewport_event
        self.vp.addEventCallback(self._vp_callback)
        hud = json.loads(json.dumps(HUD))
        for row in hud["rows"]:
            if "actions" in row:
                row["key"] = su.hudModeHotkeyRefs([_hotkey(self.state_name, a) for a in row.pop("actions")])
        next(r for r in hud["rows"] if r.get("id") == "k_create")["key"] = {
            "ctrl": "Ctrl", "shift": "Shift", "ctrlshift": "Ctrl + Shift"}[self.node.parm("createmod").evalAsString()] + " + LMB"
        self.sv.hudInfo(template=hud, update_hotkey_bindings=True)
        cam = self.pm.target_camera(self.node)
        self.pm.match_resolution(self.node, cam)
        self._warn_camera(cam)
        self._update_hud(force=True)

    def onExit(self, kwargs):
        # Inactive first: this exit fires the viewport's CameraSwitched event itself, and a watchdog
        # queued by it (or anything else deferred) must not re-attach the view after the tool is gone.
        self.active = False
        try:
            self.vp.removeEventCallback(self._vp_callback)
        except (hou.Error, AttributeError):
            pass
        try:
            self._end_drag(commit=True)
            self._set_view(0.5, 0.5, 1.0)
        except hou.ObjectWasDeleted:                    # the matcher node was deleted
            pass
        s = self.saved or {}
        self.vp.settings().setVisibleObjects(s.get("mask", "*"))
        try:
            with hou.undos.disabler():
                self.node.parm("view_mask").set("")
        except hou.ObjectWasDeleted:
            pass
        try:
            prev = s.get("camera")
            if prev is not None and prev == self._proxy():        # entered while already in the tool view
                prev = self.pm.target_camera(self.node)
            if prev is not None:
                self.vp.setCamera(prev)
            else:
                self.vp.useDefaultCamera()
                if s.get("view") is not None:
                    self.vp.setDefaultCamera(s["view"])
        except hou.ObjectWasDeleted:
            self.vp.useDefaultCamera()
        self.vp.lockCameraToView(bool(s.get("linked")))

    def onInterrupt(self, kwargs):
        self._end_drag(commit=True)
        self.pan = None

    def onResume(self, kwargs):
        self._attach()

    # ---------------------------------------------------------------- view lock + 2D pan/zoom
    def _proxy(self):
        return self.node.node("view_proxy")

    def _attach(self):
        """Look through the view proxy (= target camera + 2D window); never linked. A reference mesh
        is hidden here because the tool draws it; point clouds and splats are left to the viewport."""
        if not self.active:
            return
        cam, proxy = self.pm.target_camera(self.node), self._proxy()
        mask = self.saved["mask"] if self.saved else "*"
        ref = self._ref()
        self.vp.settings().setVisibleObjects("%s ^%s" % (mask, ref.path) if ref and ref.kind == "mesh" else mask)
        if cam is None or proxy is None:
            return
        if self.vp.camera() != proxy:
            self.vp.setCamera(proxy)
        if self.vp.isCameraLockedToView():
            self.vp.lockCameraToView(False)

    def _sync_view(self):
        """Push the (expression-driven) proxy into the view now. The viewport picks up camera
        changes asynchronously, which would leave the drawables a redraw ahead of the view."""
        proxy = self._proxy()
        if proxy is not None and self.vp.camera() == proxy:
            self.vp.setCamera(proxy)

    def _watchdog(self):
        self.watch_pending = False
        if self.pm.target_camera(self.node) is not None and self.vp.camera() != self._proxy():
            self._attach()

    def _on_viewport_event(self, **kwargs):
        # Our own setCamera calls (_sync_view, every live solve) fire this too: keep one check pending.
        if kwargs.get("event_type") == hou.geometryViewportEvent.CameraSwitched and not self.watch_pending:
            self.watch_pending = True
            _later(self._watchdog)                       # never re-enter the viewport from its callback

    def _view(self):
        n = self.node
        return n.parm("view_cx").eval(), n.parm("view_cy").eval(), n.parm("view_zoom").eval()

    def _set_view(self, cx, cy, zoom):
        zoom = min(max(zoom, ZOOM_RANGE[0]), ZOOM_RANGE[1])
        with hou.undos.disabler():
            for name, value in (("view_cx", cx), ("view_cy", cy), ("view_zoom", zoom)):
                self.node.parm(name).set(value)
        self._sync_view()
        self.affine = None              # the uv -> pixel map changed; re-measured on next use (view is in sync)

    def _fit_near(self, fr, ref):
        """Near clip of the tool view from the reference's distance and size. Cameras often keep a
        near clip of 0.001, which leaves too little depth precision to draw far or large scans as
        hidden line. Only the view proxy's clip changes, never the target camera's."""
        c = np.array(ref.center * ref.obj.worldTransform())
        want = NEAR_FRACTION * (np.linalg.norm(fr.W[3, :3] - c) + ref.radius)
        have = self.node.parm("view_near").eval()
        if not 0.5 < want / max(have, 1e-12) < 2.0:
            _later(lambda: self._set_near(want))         # no scene edits while drawing

    def _set_near(self, value):
        with hou.undos.disabler():
            self.node.parm("view_near").set(value)
        self._sync_view()

    def _zoom_at(self, uv, factor):
        cx, cy, z = self._view()
        nz = min(max(z * factor, ZOOM_RANGE[0]), ZOOM_RANGE[1])
        k = z / nz                                   # keep the image point under the cursor fixed
        self._set_view(uv[0] + (cx - uv[0]) * k, uv[1] + (cy - uv[1]) * k, nz)

    # ---------------------------------------------------------------- geometry helpers
    def _frame(self):
        """Target camera snapshot: cam, rig, W (world matrix), f (focal), near/far drawing depths
        (inside the tool view's clip range, which is the proxy's)."""
        cam = self.pm.target_camera(self.node)
        if cam is None:
            return None
        rig, proxy = self.pm.Rig(cam), self._proxy()
        near, far = (cam if proxy is None else proxy).parm("near").eval(), cam.parm("far").eval()
        return types.SimpleNamespace(cam=cam, rig=rig, W=rig.world(rig.q), f=rig.q[6],
                                     near=near * 4.0, far=near + (far - near) * 0.5)

    def _measure_frame(self, fr):
        """Where the camera frame sits in the viewport: uv -> pixels is affine (o + u*ex + v*ey) and
        depends only on viewport size, resolution and the 2D window - not on the camera pose.
        Measured while drawing (view and camera in sync) and reused by the event handlers, so live
        solves never see a view that lags the camera by a redraw."""
        P = fr.rig.unproject(fr.W, fr.f, [(0, 0), (1, 0), (0, 1)], fr.near)
        s = np.array([tuple(self.vp.mapToScreen(_vec(p))) for p in P])
        self.affine = (s[0], s[1] - s[0], s[2] - s[0])

    def _screen(self, fr, uv):
        """Camera NDC -> viewport pixels."""
        if getattr(self, "affine", None) is None:
            self._measure_frame(fr)
        o, ex, ey = self.affine
        uv = np.asarray(uv, float).reshape(-1, 2)
        return o + uv[:, :1] * ex + uv[:, 1:] * ey

    def _uv(self, fr, px):
        self._screen(fr, [(0, 0)])
        o, ex, ey = self.affine
        return np.linalg.solve(np.column_stack([ex, ey]), np.asarray(px, float) - o)

    def _mouse(self, ui_event, fr):
        """-> (uv under the cursor in camera NDC, cursor position in viewport pixels)."""
        dev = ui_event.device()
        p = hou.Vector4(dev.mouseX(), dev.mouseY(), 0.0, 1.0) * self.vp.windowToViewportTransform()
        px = np.array([p[0], p[1]])
        return self._uv(fr, px), px

    def _ray(self, fr, uv):
        """World ray through image point uv of the target camera."""
        o = fr.W[3, :3]
        return hou.Vector3(o), hou.Vector3(fr.rig.unproject(fr.W, fr.f, [uv], 1.0)[0] - o).normalized()

    def _pins(self):
        return self.drag["data"] if self.drag else self.pm.load_pins(self.node)

    def _pick(self, fr, mouse_px, pins):
        if not pins:
            return None
        d = np.linalg.norm(self._screen(fr, [p["uv"] for p in pins]) - mouse_px, axis=1)
        i = int(np.argmin(d))
        return pins[i] if d[i] <= PICK_PX else None

    def _ref(self):
        """The reference object's display geometry, prepared once per SOP cook: `kind` is 'mesh'
        (drawn by the tool; packed prims and polysoups converted to polygons), 'splat' (points with
        Houdini's GSplat attributes) or 'points'; the last two are drawn by the viewport itself.
        Picking data and the shaded copy are made on first use (_points, _shade)."""
        obj = self.pm.reference_object(self.node)
        if obj is None:
            return None
        sop = obj.displayNode()
        geo = sop.geometry()
        key = (sop.path(), sop.cookCount())
        r = self.caches.get("ref")
        if r is None or r.key != key:
            cloud = geo.intrinsicValue("primitivecount") == 0
            kind = "splat" if cloud and geo.findPointAttrib("GS_Alpha") is not None else "points"
            if not cloud:
                geo = self.pm.polygons(geo)
                try:
                    self.d_wire.setGeometry(geo)
                    self.d_face.setGeometry(geo)
                    kind = "mesh"
                except hou.OperationFailed:      # nothing the drawables can show: viewport draws it, picked as points
                    pass
            bb = geo.boundingBox()
            r = types.SimpleNamespace(key=key, path=obj.path(), kind=kind, geo=geo, center=bb.center(),
                                      radius=0.5 * bb.sizevec().length(), P=None, sigma=None, alpha=None,
                                      orient=None, shaded=False,
                                      count=geo.intrinsicValue("primitivecount" if kind == "mesh" else "pointcount"),
                                      raw_splat=cloud and kind == "points" and geo.findPointAttrib("opacity") is not None)
            prev = self.caches.get("ref")
            self.caches["ref"] = r
            if prev is not None and (prev.path, prev.kind) != (r.path, r.kind):
                _later(self._attach)                              # the viewport mask depends on the kind
        r.obj = obj
        return r

    def _points(self, ref):
        """Object-space picking data: P, plus sigma / alpha / orient for splats (float32 arrays)."""
        if ref.P is None:
            g = ref.geo
            get = lambda name, n: np.frombuffer(g.pointFloatAttribValuesAsString(name), np.float32).reshape(-1, n)
            ref.P = get("P", 3)
            if ref.kind == "splat":
                try:
                    ref.sigma, ref.alpha, ref.orient = get("scale", 3), get("GS_Alpha", 1)[:, 0], get("orient", 4)
                except (hou.OperationFailed, ValueError):         # incomplete GSplat attributes: plain points
                    ref.sigma = ref.alpha = ref.orient = None
        return ref.P

    def _hit_cloud(self, ref, uv, fr):
        """Pin anchor on Gaussian splats or a point cloud: the cursor ray is composited front to back
        through the splats (pm.splat_hit). 'points' snaps to the splat centre nearest to where the ray
        turns opaque; the other modes take that point on the ray itself."""
        xf = ref.obj.worldTransform()
        inv = xf.inverted()
        o, d = self._ray(fr, uv)
        o_l = np.array(o * inv)
        d_l = np.array((o + d) * inv) - o_l
        P = self._points(ref)
        hit = self.pm.splat_hit(o_l, d_l, P, ref.sigma, ref.alpha, ref.orient,
                                px=fr.rig.win[2] / (fr.rig.scales(fr.f)[0] * fr.rig.res[0]))
        if hit is None:
            return None
        p = P[hit[1]] if self.node.parm("snap").evalAsString() == "points" else o_l + hit[0] * d_l
        return tuple(_vec(p) * xf)

    def _hit_mesh(self, uv, fr, mouse_px):
        """Anchor for a new pin under the cursor (world space), per the 'snap' parm, or None.

        points: nearest mesh point within SNAP_PX in screen space. In the wireframe display every
                point counts, as every edge is visible, and hidden points only lose ties (+3 px);
                in the hidden-line and shaded displays hidden points don't count.
        edges:  closest point on the edges of the face under the cursor, if within SNAP_PX.
        Otherwise (and in 'surface' mode) the first surface hit of the cursor ray.
        Splats and point clouds: see _hit_cloud.
        """
        ref = self._ref()
        if ref is None:
            return None
        if ref.kind != "mesh":
            return self._hit_cloud(ref, uv, fr)
        geo = ref.geo
        xf = ref.obj.worldTransform()
        inv = xf.inverted()
        mode = self.node.parm("snap").evalAsString()
        o, d = self._ray(fr, uv)
        o_l = o * inv
        pos = hou.Vector3()
        prim_num = geo.intersect(o_l, (o + d) * inv - o_l, pos, hou.Vector3(), hou.Vector3())

        def dist_px(P):
            uv_p, z = fr.rig.project(fr.W, fr.f, P)
            dd = np.linalg.norm(self._screen(fr, uv_p) - mouse_px, axis=1)
            dd[z <= 0] = np.inf
            return dd

        if mode == "points":
            M = self.pm._m(xf)
            P = self._points(ref) @ M[:3, :3] + M[3, :3]
            dist = dist_px(P)
            hidden_cost = 3.0 if self.node.parm("meshdisplay").evalAsString() == "wire" else np.inf
            best, best_rank = None, SNAP_PX
            near = np.argpartition(dist, 16)[:16] if len(dist) > 16 else np.arange(len(dist))
            for i in near[np.argsort(dist[near])]:
                if dist[i] > best_rank:
                    break
                ray = hou.Vector3(P[i]) * inv - o_l
                h = hou.Vector3()
                hidden = geo.intersect(o_l, ray.normalized(), h, hou.Vector3(), hou.Vector3()) >= 0 and \
                    (h - o_l).length() < ray.length() * (1 - 1e-5) - 1e-6
                rank = dist[i] + (hidden_cost if hidden else 0.0)
                if rank <= best_rank:
                    best, best_rank = P[i], rank
            if best is not None:
                return tuple(best)
        if prim_num < 0:
            return None
        if mode == "edges":
            prim = geo.prim(prim_num)
            pts = [v.point().position() for v in prim.vertices()]
            closed = isinstance(prim, hou.Face) and prim.isClosed()
            best = None
            for a, b in zip(pts, pts[1:] + (pts[:1] if closed else [])):
                ab = b - a
                t = min(max((pos - a).dot(ab) / max(ab.dot(ab), 1e-30), 0.0), 1.0)
                c = a + ab * t
                if best is None or (c - pos).length() < (best - pos).length():
                    best = c
            if best is not None:
                cand = np.array(best * xf)
                if dist_px(cand[None])[0] <= SNAP_PX:
                    return tuple(cand)
        return tuple(pos * xf)

    # ---------------------------------------------------------------- pins editing
    def _edit(self, label, fn):
        return self.pm.edit_pins(self.node, label, fn)

    def _with_selected(self, label, fn):
        sel = self.selected

        def edit(data):
            for p in self.pm.pins_at(data):
                if p["id"] == sel:
                    fn(p, data)
                    return True
            return False
        return sel is not None and self._edit(label, edit)

    def _delete_selected(self):
        sel = self.selected

        def edit(data):
            pins = self.pm.pins_at(data)
            pins[:] = [p for p in pins if p["id"] != sel]
        if sel is not None:
            self._edit("Delete pin %d" % sel, edit)
            self.selected = None

    # ---------------------------------------------------------------- dragging + live solve
    def _begin_drag(self, pin_id, grab_uv, data, fr, created=False):
        pin = next(p for p in self.pm.pins_at(data) if p["id"] == pin_id)
        fatal, protected = self.pm.camera_problems(fr.cam)
        self.drag = {"id": pin_id, "data": data, "offset": np.array(pin["uv"]) - grab_uv, "moved": False,
                     "rig": None if fatal else fr.rig, "anchor": fr.rig.q.copy(), "q": fr.rig.q.copy(),
                     "free": self.pm.free_mask(self.node, protected), "cam": fr.cam, "created": created,
                     "undo": False}
        if created:
            self._open_undo("Create pin %d" % pin_id)

    def _open_undo(self, label):
        if not self.drag["undo"]:
            self.sv.beginStateUndo(self.pm.UNDO_PREFIX + label)   # one undo step per drag
            self.drag["undo"] = True

    def _drag_to(self, uv):
        dr = self.drag
        self._open_undo("Drag pin %d" % dr["id"])
        pins = self.pm.pins_at(dr["data"])
        pin = next(p for p in pins if p["id"] == dr["id"])
        pin["uv"] = [float(v) for v in uv + dr["offset"]]
        dr["moved"] = True
        if dr["rig"] is not None and pin["on"]:
            self._solve(pins, dr, polish=0)

    def _solve(self, pins, dr, polish):
        act = [p for p in pins if p["on"]]
        q, info = self.pm.solve(dr["rig"], [p["p"] for p in act], [p["uv"] for p in act], dr["free"],
                                dr["anchor"], dr["q"], polish=polish, **self.pm.solver_options(self.node))
        if info["n"]:
            dr["q"] = q
            self.pm.write_camera(dr["cam"], q, dr["free"])
            self._sync_view()
        return info

    def _end_drag(self, commit=True):
        dr, self.drag = self.drag, None
        if dr is None:
            return
        try:
            if commit and dr["moved"]:
                self.pm.save_pins(self.node, dr["data"])
                pins = self.pm.pins_at(dr["data"])
                if dr["rig"] is not None and any(p["on"] for p in pins):
                    info = self._solve(pins, dr, polish=2)
                    if info["n"] and self.node.parm("autokey").eval():
                        self.pm.key_camera(dr["cam"], dr["q"], dr["free"])
            elif commit and dr["created"]:
                self.pm.save_pins(self.node, dr["data"])
        finally:
            if dr["undo"]:
                self.sv.endStateUndo()
        self._update_hud(force=True)

    # ---------------------------------------------------------------- events
    def onMouseEvent(self, kwargs):
        ev = kwargs["ui_event"]
        dev = ev.device()
        reason = ev.reason()
        if ev.curViewport().name() != self.vp.name():     # HOM viewport wrappers never compare equal
            return False                                   # another viewport of a split layout
        self._watchdog()
        fr = self._frame()
        if fr is None:
            return dev.isLeftButton()
        uv, px = self._mouse(ev, fr)

        # 2D pan with MMB: keep the grabbed image point under the cursor.
        if dev.isMiddleButton() or self.pan is not None:
            if self.pan is None:
                self.pan = uv
            elif dev.isMiddleButton():
                cx, cy, z = self._view()
                self._set_view(cx + self.pan[0] - uv[0], cy + self.pan[1] - uv[1], z)
            if not dev.isMiddleButton() or reason == R.Changed:
                self.pan = None
            return True
        if dev.isAltKey() and (dev.isLeftButton() or dev.isRightButton()):
            return True                                                  # no Alt navigation

        if self.drag is not None:
            if reason in (R.Active, R.Start) and dev.isLeftButton():
                self._drag_to(uv)
            elif reason == R.Changed or not dev.isLeftButton():
                self._end_drag(commit=True)
            return True

        if reason in (R.Start, R.Picked) and dev.isLeftButton():
            pos, when = self.last_press
            if reason == R.Picked and pos is not None and time.time() - when < 0.5 and np.linalg.norm(px - pos) < 3:
                return True                  # the click was already handled as press/release
            if reason == R.Start:
                self.last_press = (px, time.time())
            data = self.pm.load_pins(self.node)
            if self._create_modifier(dev):
                anchor = self._hit_mesh(uv, fr, px)
                if anchor is None:
                    self.sv.setPromptMessage("Camera Pin Matcher: click on the reference mesh to create a pin",
                                             hou.promptMessageType.Warning)
                    return True
                a_uv, _ = fr.rig.project(fr.W, fr.f, np.array([anchor]))
                pin = self.pm.new_pin(data, None, anchor, a_uv[0])
                self.selected = pin["id"]
                self._begin_drag(pin["id"], uv, data, fr, created=True)
                if reason == R.Picked:
                    self._end_drag(commit=True)
                return True
            hit = self._pick(fr, px, self.pm.pins_at(data))
            self.selected = hit["id"] if hit else None
            if hit and not hit["lock"] and reason == R.Start:
                self._begin_drag(hit["id"], uv, data, fr)
            self._update_hud(force=True)
            return True
        return dev.isLeftButton()      # swallow other LMB traffic (no object picking in the tool)

    def _create_modifier(self, dev):
        mode = self.node.parm("createmod").evalAsString()
        ctrl, shift = dev.isCtrlKey(), dev.isShiftKey()
        return {"ctrl": ctrl and not shift, "shift": shift and not ctrl, "ctrlshift": ctrl and shift}[mode]

    def onMouseWheelEvent(self, kwargs):
        ev = kwargs["ui_event"]
        fr = self._frame()
        if fr is not None:
            uv, _ = self._mouse(ev, fr)
            self._zoom_at(uv, 1.25 ** ev.device().mouseWheel())
        return True                                                       # never dolly the view

    def onKeyEvent(self, kwargs):
        dev = kwargs["ui_event"].device()
        key = dev.keyString().lower()
        if key in ("del", "delete", "backspace"):
            self._delete_selected()
            return True
        if key in ("space",) or key.startswith("space"):
            return True                                                   # no volatile view tool
        return False

    def onKeyTransitEvent(self, kwargs):
        return kwargs["ui_event"].device().keyString().lower() == "space"

    def onMouseDoubleClickEvent(self, kwargs):
        return True

    def onPlaybackChangeEvent(self, kwargs):
        self._end_drag(commit=True)
        self._sync_view()
        data = self.pm.load_pins(self.node)
        src = self.pm.nearest_pin_frame(data)
        if not self.pm.pins_at(data) and src is not None:
            self.sv.setPromptMessage("%s: no pins on frame %s - press C to copy the pins of frame %d"
                                     % (LABEL, self.pm.frame_key(), src))
        else:
            self.sv.clearPromptMessage()
        self._update_hud(force=True)

    def onNodeChangeEvent(self, kwargs):
        parm = kwargs.get("parm_tuple")
        name = parm.name() if parm is not None else ""
        if name in ("camera", "refgeo"):
            self._attach()
            self._warn_camera(self.pm.target_camera(self.node))
        if name in ("camera", "plate", "matchres"):
            self.pm.match_resolution(self.node, self.pm.target_camera(self.node))
        self._update_hud(force=True)

    # ---------------------------------------------------------------- menu / hotkeys
    def onMenuPreOpen(self, kwargs):
        n, items = self.node, kwargs["menu_item_states"]
        for key, parm in (("toggle_ghosts", "showghosts"), ("toggle_errors", "showerrors")):
            items.setdefault(key, {})["value"] = bool(n.parm(parm).eval())
        for name in self.pm.PARMS + ("roll",):
            items.setdefault("lock_" + name, {})["value"] = bool(n.parm("lock_" + name).eval())
        states = kwargs["menu_states"]
        states.setdefault("plate_mode", {})["value"] = n.parm("platemode").evalAsString()
        states.setdefault("mesh_display", {})["value"] = n.parm("meshdisplay").evalAsString()
        states.setdefault("snap_mode", {})["value"] = n.parm("snap").evalAsString()

    def onMenuAction(self, kwargs):
        item, n, pm = kwargs["menu_item"], self.node, self.pm
        if item == "toggle_active":
            self._with_selected("Toggle pin active", lambda p, d: p.__setitem__("on", not p["on"]))
        elif item == "toggle_lock":
            self._with_selected("Toggle pin lock", lambda p, d: p.__setitem__("lock", not p["lock"]))
        elif item == "delete_pin":
            self._delete_selected()
        elif item == "deactivate_worst":
            info = self._evaluate(pm.target_camera(n), pm.pins_at(pm.load_pins(n)))
            errors = [(e, i) for i, e in zip(info["pin_ids"], info["errors"]) if np.isfinite(e)] if info else []
            if errors:
                e, self.selected = max(errors)
                self._with_selected("Deactivate worst pin", lambda p, d: p.__setitem__("on", False))
                self.sv.setPromptMessage("%s: pin %d (%.1f px) deactivated - A turns it back on" % (LABEL, self.selected, e))
        elif item == "solve_key":
            info = pm.solve_and_key(n)
            self._sync_view()
            self.sv.setPromptMessage("Camera Pin Matcher: %s, RMS %.2f px" % (info["status"], info.get("rms", 0)))
        elif item == "copy_nearest":
            src = pm.copy_nearest_pins(n)
            self.sv.setPromptMessage("Camera Pin Matcher: " + ("copied pins from frame %d" % src if src is not None
                                     else "nothing copied (frame has pins, or no pinned frame)"))
        elif item == "delete_keys":
            cam = pm.target_camera(n)
            if cam is not None:
                with hou.undos.group(pm.UNDO_PREFIX + "Delete keys on frame"):
                    pm.delete_keys(cam)
                self._sync_view()
        elif item == "keyed_frames":
            pm.cb_keyed_frames({"node": n})
        elif item == "reset_view":
            self._set_view(0.5, 0.5, 1.0)
        elif item == "toggle_mode":
            n.parm("platemode").set(1 - n.parm("platemode").eval())
        elif item == "cycle_display":
            n.parm("meshdisplay").set((n.parm("meshdisplay").eval() + 1) % len(DISPLAYS))
        elif item in ("plate_mode", "mesh_display", "snap_mode"):
            n.parm({"plate_mode": "platemode", "mesh_display": "meshdisplay", "snap_mode": "snap"}[item]).set(kwargs[item])
        elif item in ("toggle_ghosts", "toggle_errors"):
            p = n.parm("showghosts" if item == "toggle_ghosts" else "showerrors")
            p.set(1 - p.eval())
        elif item.startswith("lock_"):
            n.parm(item).set(1 - n.parm(item).eval())
        elif item.startswith("preset_"):
            pm.cb_preset({"node": n, "parm_name": item})
        elif item in ("activate_all", "deactivate_all", "unlock_all"):
            pm.set_all(n, "lock" if item == "unlock_all" else "on", item == "activate_all")
        elif item == "delete_frame":
            pm.delete_frame_pins(n)
        elif item == "delete_all":
            pm.delete_all_pins(n)
        self._update_hud(force=True)

    # ---------------------------------------------------------------- warnings + HUD
    def _warn_camera(self, cam):
        """Say (once per problem set) which camera parms are never written, or why solving is off.
        Non-modal: viewport flash + status bar; the HUD keeps showing it."""
        fatal, protected = self.pm.camera_problems(cam)
        key = (cam.path() if cam else None, fatal, tuple(sorted(protected.items())))
        if (fatal is None and not protected) or key in self.caches.setdefault("warned", set()):
            return
        self.caches["warned"].add(key)
        msg = ("solving is disabled: %s" % fatal) if fatal else "%s: %s never overwritten (%s)" % (
            cam.name(), ", ".join(sorted(protected)), ", ".join(sorted(set(protected.values()))))
        self.sv.flashMessage("", LABEL + ": " + msg, 8.0, self.vp)
        hou.ui.setStatusMessage(LABEL + ": " + msg, hou.severityType.Warning)

    def _update_hud(self, force=False):
        vals = self._hud_values()
        if force or vals != self.hud_values:
            self.hud_values = vals
            self.sv.hudInfo(values=vals)

    def _evaluate(self, cam, pins):
        """The solver's view of the active pins at the camera as it is (a zero-iteration solve: status,
        RMS, per-pin errors and outlier flags, plus pin_ids), or None if there is nothing to solve.
        The HUD and the pin colours need it on every redraw, so it is kept until the pins, the camera
        model or the solver settings change."""
        pm = self.pm
        act = [p for p in pins if p["on"]]
        fatal, protected = pm.camera_problems(cam)
        if fatal or not act:
            return None
        rig, free, opts = pm.Rig(cam), pm.free_mask(self.node, protected), pm.solver_options(self.node)
        key = (json.dumps(act), pickle.dumps(vars(rig)), free, sorted(opts.items()))   # vars(rig): the whole camera model
        if self.caches.get("eval", (None,))[0] != key:
            _, info = pm.solve(rig, [p["p"] for p in act], [p["uv"] for p in act], free, rig.q, iters=0, **opts)
            self.caches["eval"] = (key, dict(info, pin_ids=[p["id"] for p in act]))
        return self.caches["eval"][1]

    def _hud_values(self):
        n, pm = self.node, self.pm
        cam = pm.target_camera(n)
        data = self._pins()
        pins = pm.pins_at(data)
        act = [p for p in pins if p["on"]]
        frame = int(pm.frame_key())
        keys = pm.keyed_frames(cam)
        locks = [p for p in pm.PARMS + ("roll",) if n.parm("lock_" + p).eval()]
        src = pm.nearest_pin_frame(data) if not pins else None
        vals = {"pins": "%d / %d" % (len(act), len(pins)) + ("   (C: copy from frame %d)" % src if src is not None else ""),
                "frame": "%d%s" % (frame, "  (keyed)" if frame in keys else ""),
                "keys": (", ".join(map(str, keys[:10])) + (" ..." if len(keys) > 10 else "")) or "none",
                "status": "-", "rms": "-", "outliers": "-", "lens": "-", "locks": ", ".join(locks) or "none", "ref": "-"}
        if cam is None:
            vals["status"] = "set a Target Camera"
            return vals
        fatal, protected = pm.camera_problems(cam)
        if protected:
            vals["locks"] = ", ".join(locks + ["%s (protected)" % k for k in sorted(protected)])
        rig = pm.Rig(cam)
        f = rig.q[6]
        vals["lens"] = "%.2f / %.2f deg" % (f, np.degrees(2 * np.arctan(rig.aperture * rig.win[2] / (2 * f))))
        if fatal:
            vals["status"] = "disabled: " + fatal
        elif act:
            info = self._evaluate(cam, pins)
            vals["status"] = info["status"] + (" (%d behind camera)" % info["behind"] if info.get("behind") else "")
            vals["rms"] = "%.3f px" % info["rms"] if info["n"] else "-"
            if info.get("redundant"):                   # enough pins to tell a bad one
                bad = sorted(((e, i) for i, e, o in zip(info["pin_ids"], info["errors"], info["outliers"]) if o), reverse=True)
                vals["outliers"] = ", ".join("%d (%.1f px)" % (i, e) for e, i in bad) or "none"
            if info["roll_off"]:
                vals["locks"] = vals["locks"].replace("roll", "roll (off: %s)" % info["roll_off"])
        else:
            vals["status"] = "no active pins"
        ref = self._ref()
        if ref is None:
            vals["status"] += "  |  set Reference Geometry"
        else:
            vals["ref"] = "%s  (%s)" % (ref.path.rsplit("/", 1)[-1], {
                "mesh": "mesh, %s polygons", "splat": "Gaussian splats, %s", "points": "point cloud, %s points"}[ref.kind] % _count(ref.count)) + (
                "  - add a Bake GSplats SOP to see it as splats" if ref.raw_splat else "") + (
                "  - display flag is off" if ref.kind != "mesh" and not ref.obj.isObjectDisplayed() else "")
        path = pm.plate_path(n)
        if not path or not os.path.isfile(path):
            vals["frame"] += "  (no plate)" if not path else "  (plate missing)"
        return vals

    # ---------------------------------------------------------------- drawing
    def _make_drawables(self):
        GD, T = hou.GeometryDrawable, hou.drawableGeometryType
        sv = self.sv
        self.d_plate = None           # created per plate resolution, see _plate()
        self.d_wire = GD(sv, T.Line, "cpm_wire")
        self.d_face = GD(sv, T.Face, "cpm_face")
        # marker styles like Locate/Ring ignore Cd, so there is one target drawable per pin state
        self.d_targets = {k: GD(sv, T.Point, "cpm_targets_" + k) for k in COLORS}
        self.d_selected = GD(sv, T.Point, "cpm_selected")
        self.d_anchors = GD(sv, T.Point, "cpm_anchors")
        self.d_lines = GD(sv, T.Line, "cpm_lines")
        self.d_text = hou.TextDrawable(sv, "cpm_text")
        self.drawables = [self.d_wire, self.d_face, self.d_selected, self.d_anchors, self.d_lines, self.d_text] + \
            list(self.d_targets.values())
        for d in self.drawables:
            d.setVisibleInViewport(self.vp)
            d.show(True)

    def _plate(self):
        """Load the current plate frame into the sprite drawable. False if there is no plate.

        H22 notes: sprites fed a file path are capped at 512 px, so the image goes in as an
        ImageLayer at full resolution (needs max_resolution = image size, as ints, at creation).
        loadImageDataFromFile returns linear values and layers are shown as-is -> encode to sRGB.
        """
        path = self.pm.plate_path(self.node)
        if not path or not os.path.isfile(path):
            return False
        key = (path, os.path.getmtime(path))
        if self.caches.get("plate") == key:
            return True
        try:
            w, h = (int(v) for v in hou.imageResolution(path))
            a = np.frombuffer(hou.loadImageDataFromFile(path, hou.imageDepth.Float32), np.float32).reshape(h, w, 4)
        except (hou.Error, ValueError):
            return False
        rgb = np.clip(a[..., :3], 0.0, 1.0)
        px = np.empty((h, w, 4), np.uint8)
        px[..., :3] = np.round(np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055) * 255)
        px[..., 3] = 255
        layer = hou.ImageLayer()
        layer.setChannelCount(4)
        layer.setStorageType(hou.imageLayerStorageType.Fixed8)
        layer.setBorder(hou.imageLayerBorder.Clamp)
        layer.setDataWindow(0, 0, w, h)
        layer.setDisplayWindow(0, 0, w, h)
        layer.setAllBufferElements(px.tobytes())
        if self.caches.get("plate_res") != (w, h):
            self.d_plate = hou.GeometryDrawable(self.sv, hou.drawableGeometryType.Sprite, "cpm_plate",
                                                params={"max_resolution": [w, h]})
            self.d_plate.setVisibleInViewport(self.vp)
            self.d_plate.show(True)
            self.caches["plate_res"] = (w, h)
        self.d_plate.setParams({"images": {"plate": layer}, "default": "plate"})
        self.caches["plate"], self.caches["plate_layer"] = key, layer
        return True

    def _plate_geo(self, fr, depth, alpha):
        """One world-space sprite that exactly covers the camera frame at `depth`."""
        g = hou.Geometry()
        pt = g.createPoint()
        pt.setPosition(_vec(fr.rig.unproject(fr.W, fr.f, [(0.5, 0.5)], depth)[0]))
        fx, fy = fr.rig.scales(fr.f)
        size = (fr.rig.win[2] / fx * depth, fr.rig.win[3] / fy * depth)
        for name, default, value in (("pscale", 1.0, 1.0), ("spritescale", (1.0, 1.0), size), ("Alpha", 1.0, alpha),
                                     ("Cd", (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)), ("spriteshop", "", "plate"),
                                     ("spritescreenspace", 0, 0), ("spritepreserveaspect", 0, 0)):
            g.addAttrib(hou.attribType.Point, name, default)
            pt.setAttribValue(name, value)
        return g

    def _shade(self, ref):
        """Give the face drawable a copy of the mesh with baked N.L shading in Cd (first use only)."""
        if ref.shaded:
            return
        shaded = hou.Geometry()
        hou.sopNodeTypeCategory().nodeVerb("normal").execute(shaded, [ref.geo])
        cls = "Point" if shaded.findPointAttrib("N") else "Vertex" if shaded.findVertexAttrib("N") else None
        if cls:
            N = np.frombuffer(getattr(shaded, "%sFloatAttribValuesAsString" % cls.lower())("N"), np.float32).reshape(-1, 3)
            light = np.array([0.35, 0.8, 0.5]) / np.linalg.norm([0.35, 0.8, 0.5])
            cd = (0.3 + 0.7 * np.abs(N @ light))[:, None] * np.array([0.75, 0.78, 0.85])
            shaded.addAttrib(getattr(hou.attribType, cls), "Cd", (1.0, 1.0, 1.0))
            getattr(shaded, "set%sFloatAttribValuesFromString" % cls)("Cd", cd.astype(np.float32).tobytes())
        self.d_face.setGeometry(shaded)
        ref.shaded = True

    def _draw_mesh(self, handle, fr, ref):
        """The reference mesh in the 'meshdisplay' mode. The hidden-line modes first draw the faces
        fully transparent: that writes the drawables' depth, so edges and ghost faces behind the
        surface fail the depth test while the plate still shows through. Edges and ghost faces
        are pulled toward the eye (_pull) to win against the faces they lie on."""
        n = self.node
        disp = n.parm("meshdisplay").evalAsString()
        xf = ref.obj.worldTransform()
        eye = fr.W[3, :3]
        face = {"use_cd": False, "color1": (0.0, 0.0, 0.0, 0.0), "fade_factor": 1.0, "backface_culling": False}
        if disp != "wire":
            if disp != "hidden":
                self._shade(ref)
            self.d_face.setTransform(xf)
            self.d_face.draw(handle, dict(face, use_cd=True, color1=(1.0, 1.0, 1.0, 1.0)) if disp == "shaded" else face)
            if disp == "ghost":
                self.d_face.setTransform(xf * _pull(eye, PULL))
                self.d_face.draw(handle, dict(face, use_cd=True, color1=(1.0, 1.0, 1.0, GHOST_ALPHA)))
        alpha = n.parm("wireopacity").eval()
        if alpha > 0.0:
            wc = n.parmTuple("wirecolor").eval()
            self.d_wire.setTransform(xf if disp == "wire" else xf * _pull(eye, 2 * PULL))
            self.d_wire.draw(handle, {"color1": (wc[0], wc[1], wc[2], alpha), "line_width": 1.5,
                                      "fade_factor": 1.0, "use_cd": False})

    def onDraw(self, kwargs):
        handle = kwargs["draw_handle"]
        n = self.node
        fr = self._frame()
        if fr is None:
            return
        view = self.pm._m(self.vp.viewTransform())
        if np.abs(view[:3, :3] - fr.W[:3, :3]).max() > 1e-6 or np.abs(view[3, :3] - fr.W[3, :3]).max() > 1e-6 * (
                1.0 + np.abs(fr.W[3, :3]).max()):
            _later(self._sync_view)                          # view lags the camera (undo, script, time change)
        else:
            self._measure_frame(fr)                          # only from a view in sync with the camera
        fg = n.parm("platemode").evalAsString() == "fg"
        ref = self._ref()
        if ref is not None:
            self._fit_near(fr, ref)

        # Plate behind: drawn first, far away (the reference's depth faces must not hide it). Plate in
        # front: after the mesh, just in front of the camera. Splats and point clouds are drawn by
        # the viewport, under the tool's drawables.
        plate = self._plate()
        if plate and not fg:
            self.d_plate.setGeometry(self._plate_geo(fr, fr.far, 1.0))
            self.d_plate.draw(handle, {"fade_factor": 0.0})
        if ref is not None and ref.kind == "mesh":
            self._draw_mesh(handle, fr, ref)
        if plate and fg:
            self.d_plate.setGeometry(self._plate_geo(fr, fr.near * 1.5, n.parm("opacity").eval()))
            self.d_plate.draw(handle, {"fade_factor": 1.0})

        self._draw_pins(handle, fr)
        self._update_hud()

    def _points_geo(self, P, colors=None):
        g = hou.Geometry()
        g.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
        if len(P):
            g.createPoints([tuple(map(float, p)) for p in P])
            if colors is not None:
                g.setPointFloatAttribValues("Cd", [float(c) for col in colors for c in col])
        return g

    def _markers(self, drawable, P, color, style, radius, alpha=1.0):
        if len(P):
            drawable.setGeometry(self._points_geo(P))
            drawable.draw(self._handle, {"style": style, "radius": radius, "color1": tuple(color) + (alpha,),
                                         "use_cd": False, "fade_factor": 1.0, "glow_width": 3,
                                         "highlight_mode": hou.drawableHighlightMode.MatteOverGlow,
                                         "color2": (0.0, 0.0, 0.0, 0.85 * alpha)})

    def _draw_pins(self, handle, fr):
        """Target marker (state colour) + reprojected anchor dot + residual line + id/error label per
        pin; ghost targets of the previous/next pinned frame."""
        n, pm = self.node, self.pm
        rig, W, f, depth = fr.rig, fr.W, fr.f, fr.near
        self._handle = handle
        S = hou.drawableGeometryPointStyle
        data = self._pins()
        pins = pm.pins_at(data)
        show_err = n.parm("showerrors").eval()
        text = []
        if pins:
            uv_t = np.array([p["uv"] for p in pins])
            uv_a, z = rig.project(W, f, np.array([p["p"] for p in pins]))
            front = z > 0
            err = rig.pixel_errors(uv_a, uv_t)
            info = self._evaluate(fr.cam, pins)
            bad = {i for i, o in zip(info["pin_ids"], info["outliers"]) if o} if info else set()
            state = ["inactive" if not p["on"] else "outlier" if p["id"] in bad else "locked" if p["lock"] else "active"
                     for p in pins]
            T = rig.unproject(W, f, uv_t, depth)
            A = rig.unproject(W, f, uv_a[front], depth)
            for key in ("inactive", "locked", "active", "outlier"):
                self._markers(self.d_targets[key], T[[s_ == key for s_ in state]], COLORS[key], S.Locate, 15)
            sel = [p["id"] == self.selected for p in pins]
            self._markers(self.d_selected, T[sel], COLORS["selected"], S.Ring, 24)
            cols = [COLORS[s_] for s_, fr_ in zip(state, front) if fr_]
            self.d_anchors.setGeometry(self._points_geo(A, cols))
            self.d_anchors.draw(handle, {"style": S.SmoothCircle, "radius": 6, "fade_factor": 1.0, "glow_width": 2,
                                         "highlight_mode": hou.drawableHighlightMode.MatteOverGlow,
                                         "color2": (0.0, 0.0, 0.0, 0.85)})
            lg = self._points_geo(np.stack([T[front], A], 1).reshape(-1, 3), np.repeat(cols, 2, axis=0))
            lg.createPolygons([(i, i + 1) for i in range(0, 2 * len(A), 2)], False)   # open: target -> anchor (no keyword)
            self.d_lines.setGeometry(lg)
            self.d_lines.draw(handle, {"line_width": 2.0, "fade_factor": 1.0})
            for p, sc, e, st, fr_ in zip(pins, self._screen(fr, uv_t), err, state, front):
                label = str(p["id"]) + (("  %.2f px" % e) if show_err and fr_ else "") + ("" if fr_ else "  (behind camera)")
                text.append((label, sc, COLORS["selected"] if p["id"] == self.selected else COLORS[st]))
        if n.parm("showghosts").eval():
            gp, gid = [], []
            for fr_ in pm.neighbour_frames(data):
                for p in pm.pins_at(data, fr_) if fr_ is not None else []:
                    gp.append(p["uv"])
                    gid.append("%d'  (f%d)" % (p["id"], fr_))
            if gp:
                self._markers(self.d_targets["ghost"], rig.unproject(W, f, gp, depth), COLORS["ghost"], S.Locate, 11, 0.7)
                text += [(label, sc, COLORS["ghost"]) for label, sc in zip(gid, self._screen(fr, gp))]
        for label, sc, c in text:     # hou.Color renders crisp text (tuples come out dim); glow blurs it
            self.d_text.draw(handle, {"text": label, "translate": hou.Vector3(sc[0] + 12, sc[1] + 10, 0),
                                      "color1": hou.Color(*(0.35 * v + 0.65 for v in c)),
                                      "scale": hou.Vector3(1.25, 1.25, 1.25), "glow_width": 0,
                                      "highlight_mode": hou.drawableHighlightMode.Matte})


def _menu(state_name, definitions):
    cat = hou.objNodeTypeCategory()
    hk = {k: su.defineHotkey(definitions, state_name, k, key, label, label, cat) for k, (label, key) in ACTIONS.items()}
    m = hou.ViewerStateMenu(state_name + "_menu", LABEL)
    m.addActionItem("toggle_active", "Toggle Pin Active", hk["toggle_active"])
    m.addActionItem("toggle_lock", "Toggle Pin Lock", hk["toggle_lock"])
    m.addActionItem("delete_pin", "Delete Pin", hk["delete_pin"])
    m.addActionItem("deactivate_worst", "Deactivate Worst Pin", hk["deactivate_worst"])
    m.addSeparator()
    m.addActionItem("solve_key", "Solve & Key", hk["solve_key"])
    m.addActionItem("copy_nearest", "Copy Pins from Nearest Keyed Frame", hk["copy_nearest"])
    m.addActionItem("delete_keys", "Delete Keys on Current Frame")
    m.addActionItem("keyed_frames", "Keyed Frames...")
    m.addSeparator()
    m.addRadioStrip("plate_mode", "Plate Mode", "bg")
    m.addRadioStripItem("plate_mode", "bg", "Plate Behind Geometry")
    m.addRadioStripItem("plate_mode", "fg", "Plate Over Geometry")
    m.addActionItem("toggle_mode", "Toggle Plate Mode", hk["toggle_mode"])
    display = hou.ViewerStateMenu("display_menu", "Mesh Display")
    display.addRadioStrip("mesh_display", "Mesh Display", "hidden")
    for k, lbl in DISPLAYS:
        display.addRadioStripItem("mesh_display", k, lbl)
    display.addActionItem("cycle_display", "Cycle Mesh Display", hk["cycle_display"])
    m.addMenu(display)
    m.addToggleItem("toggle_ghosts", "Show Ghost Pins", True, hk["toggle_ghosts"])
    m.addToggleItem("toggle_errors", "Show Pin Errors", True)
    m.addActionItem("reset_view", "Reset 2D View", hk["reset_view"])
    snap = hou.ViewerStateMenu("snap_menu", "Snap New Pins To")
    snap.addRadioStrip("snap_mode", "Snap", "points")
    for k, lbl in (("points", "Points"), ("edges", "Edges"), ("surface", "Free Surface Hit")):
        snap.addRadioStripItem("snap_mode", k, lbl)
    m.addMenu(snap)
    m.addSeparator()
    locks = hou.ViewerStateMenu("lock_menu", "Lock Camera Parameters")
    for name in ("tx", "ty", "tz", "rx", "ry", "rz", "focal", "roll"):
        locks.addToggleItem("lock_" + name, name.upper() if name not in ("focal", "roll") else name.title(), False)
    locks.addSeparator()
    for k, lbl in (("preset_focal", "Preset: Lock Focal/FOV"), ("preset_nodal", "Preset: Nodal (Lock Position)"),
                   ("preset_roll", "Preset: Lock Roll"), ("preset_clear", "Unlock All Parameters")):
        locks.addActionItem(k, lbl)
    m.addMenu(locks)
    pins = hou.ViewerStateMenu("pins_menu", "All Pins")
    for k, lbl in (("activate_all", "Activate All Pins"), ("deactivate_all", "Deactivate All Pins"),
                   ("unlock_all", "Unlock All Pins"), ("delete_frame", "Delete Pins on Current Frame"),
                   ("delete_all", "Delete All Pins...")):
        pins.addActionItem(k, lbl)
    m.addMenu(pins)
    return m


def createViewerStateTemplate():
    state_name = kwargs["type"].definition().sections()["DefaultState"].contents()  # noqa: F821 (Houdini global)
    template = hou.ViewerStateTemplate(state_name, LABEL, hou.objNodeTypeCategory())
    template.bindFactory(State)
    template.bindIcon(kwargs["type"].icon())  # noqa: F821
    definitions = hou.PluginHotkeyDefinitions()
    template.bindMenu(_menu(state_name, definitions))
    template.bindHotkeyDefinitions(definitions)
    template.bindPlaybackChangeEvent()
    template.bindNodeParmChangeEvent(["camera", "refgeo", "plate", "matchres"])
    return template
