"""Camera Pin Matcher - viewer state (HDA ViewerStateModule).

Locks the viewport to the target camera (through the asset's view proxy camera, which mirrors
the target camera exactly and carries the 2D pan/zoom in its own screen window), draws the plate,
the reference mesh and the pins with drawables, and turns pin edits into live camera solves.
Math, storage and camera IO live in the asset's PythonModule (node.hdaModule()).
"""
import os
import types

import numpy as np

import hdefereval
import hou
import viewerstate.utils as su

LABEL = "Camera Pin Matcher"
PICK_PX = 12.0
ZOOM_RANGE = (0.25, 64.0)
COLORS = {"active": (0.15, 1.0, 0.35), "inactive": (0.6, 0.6, 0.6), "locked": (1.0, 0.55, 0.1),
          "selected": (1.0, 0.95, 0.15), "ghost": (0.8, 0.4, 1.0)}
R = hou.uiEventReason

HUD = {
    "title": LABEL, "desc": "tool", "icon": "OBJ_camera",
    "rows": [
        {"id": "pins", "label": "Active pins"},
        {"id": "frame", "label": "Frame"},
        {"id": "status", "label": "Solver"},
        {"id": "rms", "label": "RMS error"},
        {"id": "locks", "label": "Locked"},
        {"id": "lens", "label": "Focal / H-FOV"},
        {"id": "keys", "label": "Keyed frames"},
        {"type": "divider"},
        {"id": "k_create", "label": "Create pin on mesh", "key": "Ctrl + LMB"},
        {"label": "Select / drag pin", "key": "LMB"},
        {"label": "Delete selected pin", "key": "del / backspace"},
        {"label": "Toggle active / locked", "key": "A / L"},
        {"label": "Solve & key", "key": "K"},
        {"label": "Copy pins from nearest keyed frame", "key": "C"},
        {"label": "2D pan / zoom", "key": "MMB / mousewheel"},
        {"label": "Reset 2D view", "key": "H"},
        {"label": "Plate mode / ghosts", "key": "M / G"},
    ]}

# Hotkeyable menu actions: id -> (label, default key)
ACTIONS = {
    "toggle_active": ("Toggle Pin Active", "A"), "toggle_lock": ("Toggle Pin Lock", "L"),
    "solve_key": ("Solve & Key", "K"), "copy_nearest": ("Copy Pins from Nearest Keyed Frame", "C"),
    "reset_view": ("Reset 2D View", "H"), "toggle_mode": ("Toggle Plate Mode", "M"),
    "toggle_ghosts": ("Show Ghost Pins", "G"),
}


def _vec(v):
    return hou.Vector3(float(v[0]), float(v[1]), float(v[2]))


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
        self.caches = {}

    # ---------------------------------------------------------------- lifecycle
    def onEnter(self, kwargs):
        self.node = kwargs["node"]
        self.pm = self.node.hdaModule()
        kwargs["state_flags"]["indirect_handle_drag"] = False     # MMB is ours (2D pan)
        self.vp = self.sv.curViewport()
        self._make_drawables()
        settings = self.vp.settings()
        self.saved = {"camera": self.vp.camera(), "linked": self.vp.isCameraLockedToView(),
                      "view": self.vp.defaultCamera().stash(), "mask": settings.visibleObjects()}
        self._set_view(0.5, 0.5, 1.0)
        self._attach()
        self._vp_callback = self._on_viewport_event
        self.vp.addEventCallback(self._vp_callback)
        self.sv.hudInfo(template=HUD)
        cam = self.pm.target_camera(self.node)
        self.pm.match_resolution(self.node, cam)
        self._warn_camera(cam)
        self._update_hud(force=True)

    def onExit(self, kwargs):
        self._end_drag(commit=True)
        try:
            self.vp.removeEventCallback(self._vp_callback)
        except (hou.Error, AttributeError):
            pass
        s = self.saved or {}
        self.vp.settings().setVisibleObjects(s.get("mask", "*"))
        self._set_view(0.5, 0.5, 1.0)
        if s.get("camera") is not None:
            self.vp.setCamera(s["camera"])
        else:
            self.vp.useDefaultCamera()
            if s.get("view") is not None:
                self.vp.setDefaultCamera(s["view"])
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
        """Look through the view proxy (= target camera + 2D window); never linked."""
        cam, proxy = self.pm.target_camera(self.node), self._proxy()
        mask = self.saved["mask"] if self.saved else "*"
        ref = self.pm.reference_object(self.node)
        self.vp.settings().setVisibleObjects(mask if ref is None else "%s ^%s" % (mask, ref.path()))
        if cam is None or proxy is None:
            return
        if self.vp.camera() != proxy:
            self.vp.setCamera(proxy)
        if self.vp.isCameraLockedToView():
            self.vp.lockCameraToView(False)

    def _watchdog(self):
        if self.pm.target_camera(self.node) is not None and self.vp.camera() != self._proxy():
            self._attach()

    def _on_viewport_event(self, **kwargs):
        if kwargs.get("event_type") == hou.geometryViewportEvent.CameraSwitched:
            hdefereval.executeDeferred(self._watchdog)   # never re-enter the viewport from its callback

    def _view(self):
        n = self.node
        return n.parm("view_cx").eval(), n.parm("view_cy").eval(), n.parm("view_zoom").eval()

    def _set_view(self, cx, cy, zoom):
        zoom = min(max(zoom, ZOOM_RANGE[0]), ZOOM_RANGE[1])
        with hou.undos.disabler():
            for name, value in (("view_cx", cx), ("view_cy", cy), ("view_zoom", zoom)):
                self.node.parm(name).set(value)

    def _zoom_at(self, uv, factor):
        cx, cy, z = self._view()
        nz = min(max(z * factor, ZOOM_RANGE[0]), ZOOM_RANGE[1])
        k = z / nz                                   # keep the image point under the cursor fixed
        self._set_view(uv[0] + (cx - uv[0]) * k, uv[1] + (cy - uv[1]) * k, nz)

    # ---------------------------------------------------------------- geometry helpers
    def _frame(self):
        """Target camera snapshot: cam, rig, W (world matrix), f (focal), near/far drawing depths."""
        cam = self.pm.target_camera(self.node)
        if cam is None:
            return None
        rig = self.pm.Rig(cam)
        near, far = cam.parm("near").eval(), cam.parm("far").eval()
        return types.SimpleNamespace(cam=cam, rig=rig, W=rig.world(rig.q), f=rig.q[6],
                                     near=near * 4.0, far=near + (far - near) * 0.5)

    def _screen(self, fr, uv):
        """Camera NDC -> viewport pixels (through whatever window the viewport shows)."""
        P = fr.rig.unproject(fr.W, fr.f, uv, fr.near)
        return np.array([tuple(self.vp.mapToScreen(_vec(p))) for p in P]).reshape(-1, 2)

    def _mouse(self, ui_event, fr):
        """-> (uv under the cursor in camera NDC, cursor position in viewport pixels)."""
        o, d = ui_event.ray()
        p = np.array(o) + np.array(d) * 10.0 * max(1.0, np.linalg.norm(np.array(o) - fr.W[3, :3]))
        uv, _ = fr.rig.project(fr.W, fr.f, p[None])
        return uv[0], self._screen(fr, uv)[0]

    def _pins(self):
        return self.drag["data"] if self.drag else self.pm.load_pins(self.node)

    def _pick(self, fr, mouse_px, pins):
        if not pins:
            return None
        d = np.linalg.norm(self._screen(fr, [p["uv"] for p in pins]) - mouse_px, axis=1)
        i = int(np.argmin(d))
        return pins[i] if d[i] <= PICK_PX else None

    def _hit_mesh(self, ui_event, mouse_px):
        """Ray-cast the reference geometry, snap per the 'snap' parm. -> world anchor or None."""
        obj = self.pm.reference_object(self.node)
        if obj is None:
            return None
        geo = obj.displayNode().geometry()
        xf = obj.worldTransform()
        inv = xf.inverted()
        o, d = ui_event.ray()
        o_l = o * inv
        d_l = (o + d) * inv - o_l
        pos, nml, uvw = hou.Vector3(), hou.Vector3(), hou.Vector3()
        prim_num = geo.intersect(o_l, d_l, pos, nml, uvw)
        if prim_num < 0:
            return None
        mode = self.node.parm("snap").evalAsString()
        prim = geo.prim(prim_num)
        pts = [v.point().position() for v in prim.vertices()]
        if mode == "points" and pts:
            world = [p * xf for p in pts]
            scr = np.array([tuple(self.vp.mapToScreen(p)) for p in world])
            return tuple(world[int(np.argmin(np.linalg.norm(scr - mouse_px, axis=1)))])
        if mode == "edges" and len(pts) > 1:
            best, best_d = pos, float("inf")
            closed = prim.type() == hou.primType.Polygon and prim.isClosed()
            for a, b in zip(pts, pts[1:] + (pts[:1] if closed else [])):
                ab = b - a
                t = min(max((pos - a).dot(ab) / max(ab.dot(ab), 1e-30), 0.0), 1.0)
                c = a + ab * t
                if (c - pos).length() < best_d:
                    best, best_d = c, (c - pos).length()
            pos = best
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
            data = self.pm.load_pins(self.node)
            if self._create_modifier(dev):
                anchor = self._hit_mesh(ev, px)
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
        states.setdefault("snap_mode", {})["value"] = n.parm("snap").evalAsString()

    def onMenuAction(self, kwargs):
        item, n, pm = kwargs["menu_item"], self.node, self.pm
        if item == "toggle_active":
            self._with_selected("Toggle pin active", lambda p, d: p.__setitem__("on", not p["on"]))
        elif item == "toggle_lock":
            self._with_selected("Toggle pin lock", lambda p, d: p.__setitem__("lock", not p["lock"]))
        elif item == "delete_pin":
            self._delete_selected()
        elif item == "solve_key":
            info = pm.solve_and_key(n)
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
        elif item == "keyed_frames":
            pm.cb_keyed_frames({"node": n})
        elif item == "reset_view":
            self._set_view(0.5, 0.5, 1.0)
        elif item == "toggle_mode":
            n.parm("platemode").set(1 - n.parm("platemode").eval())
        elif item in ("plate_mode", "snap_mode"):
            n.parm("platemode" if item == "plate_mode" else "snap").set(kwargs[item])
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
        fatal, protected = self.pm.camera_problems(cam)
        key = (cam.path() if cam else None, fatal, tuple(sorted(protected.items())))
        if (fatal is None and not protected) or key in self.caches.setdefault("warned", set()):
            return
        self.caches["warned"].add(key)
        lines = (["Solving is disabled: %s." % fatal] if fatal else []) + \
                ["%s: %s - treated as locked, never overwritten." % (k, v) for k, v in sorted(protected.items())]
        hou.ui.displayMessage("Camera %s\n\n%s" % (cam.path() if cam else "(none)", "\n".join(lines)),
                              severity=hou.severityType.Warning, title=LABEL)

    def _update_hud(self, force=False):
        vals = self._hud_values()
        if force or vals != self.hud_values:
            self.hud_values = vals
            self.sv.hudInfo(hud_values=vals)

    def _hud_values(self):
        n, pm = self.node, self.pm
        cam = pm.target_camera(n)
        data = self._pins()
        pins = pm.pins_at(data)
        act = [p for p in pins if p["on"]]
        frame = int(pm.frame_key())
        keys = pm.keyed_frames(cam)
        locks = [p for p in pm.PARMS + ("roll",) if n.parm("lock_" + p).eval()]
        vals = {"pins": "%d / %d" % (len(act), len(pins)), "frame": "%d%s" % (frame, "  (keyed)" if frame in keys else ""),
                "keys": (", ".join(map(str, keys[:10])) + (" ..." if len(keys) > 10 else "")) or "none",
                "status": "-", "rms": "-", "lens": "-", "locks": ", ".join(locks) or "none"}
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
            _, info = pm.solve(rig, [p["p"] for p in act], [p["uv"] for p in act], pm.free_mask(n, protected),
                               rig.q, iters=0, **pm.solver_options(n))
            vals["status"] = info["status"] + (" (%d behind camera)" % info["behind"] if info.get("behind") else "")
            vals["rms"] = "%.3f px" % info["rms"] if info["n"] else "-"
        else:
            vals["status"] = "no active pins"
        if pm.reference_object(n) is None:
            vals["status"] += "  |  set Reference Geometry"
        return vals

    # ---------------------------------------------------------------- drawing
    def _make_drawables(self):
        GD, T = hou.GeometryDrawable, hou.drawableGeometryType
        sv = self.sv
        self.d_plate = None           # created per plate resolution, see _plate()
        self.d_wire = GD(sv, T.Line, "cpm_wire")
        self.d_face = GD(sv, T.Face, "cpm_face")
        self.d_targets = GD(sv, T.Point, "cpm_targets")
        self.d_selected = GD(sv, T.Point, "cpm_selected")
        self.d_anchors = GD(sv, T.Point, "cpm_anchors")
        self.d_lines = GD(sv, T.Line, "cpm_lines")
        self.d_ghosts = GD(sv, T.Point, "cpm_ghosts")
        self.d_text = hou.TextDrawable(sv, "cpm_text")
        self.drawables = [self.d_wire, self.d_face, self.d_targets, self.d_selected,
                          self.d_anchors, self.d_lines, self.d_ghosts, self.d_text]
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

    def _mesh(self):
        """(object, world xform, display geometry, shaded copy) cached per cook/frame."""
        obj = self.pm.reference_object(self.node)
        if obj is None:
            return None
        sop = obj.displayNode()
        key = (sop.path(), sop.cookCount(), hou.frame())
        c = self.caches.get("mesh")
        if c is None or c[0] != key:
            geo = sop.geometry()
            shaded = hou.Geometry()
            hou.sopNodeTypeCategory().nodeVerb("normal").execute(shaded, [geo])
            if shaded.findPointAttrib("N") or shaded.findVertexAttrib("N"):
                cls = "Point" if shaded.findPointAttrib("N") else "Vertex"
                N = np.array(getattr(shaded, "%sFloatAttribValues" % cls.lower())("N")).reshape(-1, 3)
                light = np.array([0.35, 0.8, 0.5]) / np.linalg.norm([0.35, 0.8, 0.5])
                shade = 0.3 + 0.7 * np.abs(N @ light)
                cd = np.repeat(shade[:, None], 3, 1) * np.array([0.75, 0.78, 0.85])
                shaded.addAttrib(getattr(hou.attribType, cls), "Cd", (1.0, 1.0, 1.0))
                getattr(shaded, "set%sFloatAttribValues" % cls)("Cd", cd.ravel().tolist())
            self.d_wire.setGeometry(geo)
            self.d_face.setGeometry(shaded)
            c = self.caches["mesh"] = (key, geo, shaded)
        return obj

    def onDraw(self, kwargs):
        handle = kwargs["draw_handle"]
        n, pm = self.node, self.pm
        fr = self._frame()
        if fr is None:
            return
        fg = n.parm("platemode").evalAsString() == "fg"
        opacity = n.parm("opacity").eval()

        # reference mesh
        obj = self._mesh()
        if obj is not None:
            xf = obj.worldTransform()
            if fg:
                self.d_face.setTransform(xf)
                self.d_face.draw(handle, {"use_cd": True, "fade_factor": 1.0, "backface_culling": False})
            else:
                self.d_wire.setTransform(xf)
                wc = n.parmTuple("wirecolor").eval()
                self.d_wire.draw(handle, {"color1": hou.Vector4(wc[0], wc[1], wc[2], 1.0), "line_width": 1.5,
                                          "fade_factor": 1.0, "use_cd": False})

        # plate (background far away, or foreground right in front of the camera)
        if self._plate():
            depth = fr.near * 1.5 if fg else fr.far
            self.d_plate.setGeometry(self._plate_geo(fr, depth, opacity if fg else 1.0))
            self.d_plate.draw(handle, {"fade_factor": 1.0 if fg else 0.0})

        self._draw_pins(handle, fr)
        self._update_hud()

    def _points_geo(self, P, colors):
        g = hou.Geometry()
        g.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
        if len(P):
            g.createPoints([tuple(map(float, p)) for p in P])
            g.setPointFloatAttribValues("Cd", [float(c) for col in colors for c in col])
        return g

    def _draw_pins(self, handle, fr):
        n, pm = self.node, self.pm
        rig, W, f, depth = fr.rig, fr.W, fr.f, fr.near
        data = self._pins()
        pins = pm.pins_at(data)
        show_err = n.parm("showerrors").eval()
        text = []
        if pins:
            uv_t = np.array([p["uv"] for p in pins])
            uv_a, z = rig.project(W, f, np.array([p["p"] for p in pins]))
            front = z > 0
            err = rig.pixel_errors(uv_a, uv_t)
            cols = [COLORS["selected"] if p["id"] == self.selected else COLORS["inactive"] if not p["on"]
                    else COLORS["locked"] if p["lock"] else COLORS["active"] for p in pins]
            T = rig.unproject(W, f, uv_t, depth)
            A = rig.unproject(W, f, uv_a[front], depth)
            self.d_targets.setGeometry(self._points_geo(T, cols))
            self.d_targets.draw(handle, {"style": hou.drawableGeometryPointStyle.RingsCircle, "radius": 7,
                                         "num_rings": 2, "falloff_range": hou.Vector2(0, 1), "fade_factor": 1.0})
            sel = [i for i, p in enumerate(pins) if p["id"] == self.selected]
            if sel:
                self.d_selected.setGeometry(self._points_geo(T[sel], [COLORS["selected"]]))
                self.d_selected.draw(handle, {"style": hou.drawableGeometryPointStyle.RingsCircle, "radius": 12,
                                              "num_rings": 1, "falloff_range": hou.Vector2(0, 1), "fade_factor": 1.0})
            self.d_anchors.setGeometry(self._points_geo(A, [c for c, fr in zip(cols, front) if fr]))
            self.d_anchors.draw(handle, {"style": hou.drawableGeometryPointStyle.SmoothSquare, "radius": 3,
                                         "fade_factor": 1.0})
            lg = hou.Geometry()
            lg.addAttrib(hou.attribType.Point, "Cd", (1.0, 1.0, 1.0))
            for (t, a, c) in zip(T[front], A, [c for c, fr in zip(cols, front) if fr]):
                poly = lg.createPolygon(is_closed=False)
                for p in (t, a):
                    pt = lg.createPoint()
                    pt.setPosition(_vec(p))
                    pt.setAttribValue("Cd", c)
                    poly.addVertex(pt)
            self.d_lines.setGeometry(lg)
            self.d_lines.draw(handle, {"line_width": 1.5, "fade_factor": 1.0})
            scr = self._screen(fr, uv_t)
            for p, s, e, c, fr in zip(pins, scr, err, cols, front):
                label = str(p["id"]) + (("  %.2fpx" % e) if show_err and fr else "") + ("" if fr else "  (behind)")
                text.append((label, s, c))
        if n.parm("showghosts").eval():
            gp, gc = [], []
            for fr in pm.neighbour_frames(data):
                if fr is not None:
                    for p in pm.pins_at(data, fr):
                        gp.append(p["uv"])
                        gc.append(p["id"])
            if gp:
                T = rig.unproject(W, f, gp, depth)
                self.d_ghosts.setGeometry(self._points_geo(T, [COLORS["ghost"]] * len(T)))
                self.d_ghosts.draw(handle, {"style": hou.drawableGeometryPointStyle.RingsCircle, "radius": 5,
                                            "num_rings": 1, "falloff_range": hou.Vector2(0, 1), "fade_factor": 1.0})
                for pid, s in zip(gc, self._screen(fr, gp)):
                    text.append(("%d'" % pid, s, COLORS["ghost"]))
        for label, s, c in text:
            self.d_text.draw(handle, {"text": label, "translate": hou.Vector3(s[0] + 9, s[1] + 6, 0),
                                      "color1": hou.Vector4(c[0], c[1], c[2], 1.0)})


def _menu(state_name, definitions):
    cat = hou.objNodeTypeCategory()
    hk = {k: su.defineHotkey(definitions, state_name, k, key, label, label, cat) for k, (label, key) in ACTIONS.items()}
    m = hou.ViewerStateMenu(state_name + "_menu", LABEL)
    m.addActionItem("toggle_active", "Toggle Pin Active", hk["toggle_active"])
    m.addActionItem("toggle_lock", "Toggle Pin Lock", hk["toggle_lock"])
    m.addActionItem("delete_pin", "Delete Pin  (Del)")
    m.addSeparator()
    m.addActionItem("solve_key", "Solve & Key", hk["solve_key"])
    m.addActionItem("copy_nearest", "Copy Pins from Nearest Keyed Frame", hk["copy_nearest"])
    m.addActionItem("delete_keys", "Delete Keys on Current Frame")
    m.addActionItem("keyed_frames", "Keyed Frames...")
    m.addSeparator()
    m.addRadioStrip("plate_mode", "Plate Mode", "bg")
    m.addRadioStripItem("plate_mode", "bg", "Plate Background, Wireframe Mesh")
    m.addRadioStripItem("plate_mode", "fg", "Plate Foreground over Shaded Mesh")
    m.addActionItem("toggle_mode", "Toggle Plate Mode", hk["toggle_mode"])
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
