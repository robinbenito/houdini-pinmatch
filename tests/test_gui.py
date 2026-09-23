"""GUI integration test for the Camera Pin Matcher viewer state.

Runs inside a graphical Houdini session:   houdini -foreground tests/test_gui.py
(or, in a session started with dev/start_rpc.py:   hython dev/rpc.py tests/test_gui.py).
It loads the test scene, enters the tool, and drives the live state's event handlers with mock
UI events whose rays come from the real viewport (GeometryViewport.mapToWorld), so everything
below the raw OS input layer is exercised: picking, snapping, live solves, drawables, parm
writes, keys, undo.
Results go to tests/gui_test_log.txt; Houdini exits when launched from the command line.
"""
import gc
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback

import numpy as np

import hdefereval
import hou

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(sys._getframe().f_code.co_filename)))  # no __file__ in houdini <script>
EXT = {hou.licenseCategoryType.Commercial: "", hou.licenseCategoryType.Indie: "lc"}.get(hou.licenseCategory(), "nc")
SCENE = os.path.join(ROOT, "scenes", "pinmatch_test.hip" + EXT)
HDA = os.path.join(ROOT, "otls", "camera_pin_matcher.hda" + EXT)
LOG = os.path.join(ROOT, "tests", "gui_test_log.txt")
R = hou.uiEventReason
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    lines.append(s)
    print(s)


# ------------------------------------------------------------------ mock UI events
class Device(object):
    def __init__(self, x, y, left=False, middle=False, right=False, ctrl=False, shift=False, alt=False,
                 wheel=0, key=""):
        self.__dict__.update(locals())

    def mouseX(self): return self.x
    def mouseY(self): return self.y
    def isLeftButton(self): return self.left
    def isMiddleButton(self): return self.middle
    def isRightButton(self): return self.right
    def isCtrlKey(self): return self.ctrl
    def isShiftKey(self): return self.shift
    def isAltKey(self): return self.alt
    def mouseWheel(self): return self.wheel
    def keyString(self): return self.key


class Event(object):
    def __init__(self, vp, dev, reason):
        self.vp, self.dev, self.r = vp, dev, reason

    def device(self): return self.dev
    def reason(self): return self.r

    def curViewport(self):                 # a fresh wrapper, like real HOM (wrappers never compare equal)
        return hou.ui.paneTabOfType(hou.paneTabType.SceneViewer).curViewport()

    def ray(self):
        d, o = self.vp.mapToWorld(self.dev.x, self.dev.y)
        return o, d


class Driver(object):
    def __init__(self, st):
        self.st, self.vp = st, st.vp

    def _send(self, x, y, reason, **kw):
        # viewport pixels -> window coordinates, as real ViewerEvents report them
        p = hou.Vector4(x, y, 0.0, 1.0) * self.vp.windowToViewportTransform().inverted()
        self.st.onMouseEvent({"ui_event": Event(self.vp, Device(p[0], p[1], **kw), reason), "state_flags": {}})

    def click(self, p, **mods):
        self._send(p[0], p[1], R.Picked, left=True, **mods)

    def drag(self, a, b, steps=12, button="left", **mods):
        btn = {button: True}
        self._send(a[0], a[1], R.Start, **btn, **mods)
        for i in range(1, steps + 1):
            t = i / float(steps)
            self._send(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, R.Active, **btn, **mods)
        self._send(b[0], b[1], R.Changed, **mods)

    def wheel(self, p, clicks):
        q = hou.Vector4(p[0], p[1], 0.0, 1.0) * self.vp.windowToViewportTransform().inverted()
        self.st.onMouseWheelEvent({"ui_event": Event(self.vp, Device(q[0], q[1], wheel=clicks), R.Located)})

    def key(self, k):
        return self.st.onKeyEvent({"ui_event": Event(self.vp, Device(0, 0, key=k), R.Picked)})

    def menu(self, item, **extra):
        self.st.onMenuAction(dict({"menu_item": item}, **extra))


# ------------------------------------------------------------------ helpers
def live_state(node):
    """The running state instance for `node` (Houdini keeps it private; find it among live objects)."""
    for o in gc.get_objects():
        try:
            if type(o).__name__ == "State" and getattr(o, "state_name", "").startswith("pinmatch::") \
                    and getattr(o, "drawables", None) and o.node.sessionId() == node.sessionId():
                return o
        except Exception:          # stale instances of deleted nodes
            pass


def refresh(st=None):
    from PySide6 import QtWidgets
    if st is not None:
        st._sync_view()          # the viewport picks up script-driven camera/time changes asynchronously
    hou.ui.paneTabOfType(hou.paneTabType.SceneViewer).curViewport().draw()
    QtWidgets.QApplication.processEvents()


def checksum(obj):
    """Everything about the reference object and its network: parms, flags, geometry, transform."""
    h = hashlib.sha1(obj.asCode(recurse=True).encode())
    g = obj.displayNode().geometry()
    h.update(np.array(g.pointFloatAttribValues("P")).tobytes())
    h.update(str([[v.point().number() for v in p.vertices()] for p in g.prims()]).encode())
    h.update(str(obj.worldTransform().asTuple()).encode())
    return h.hexdigest()


def cam_error(pm, cam, gt):
    Wc, Wg = pm._m(cam.worldTransform()), pm._m(gt.worldTransform())
    dpos = float(np.linalg.norm(Wc[3, :3] - Wg[3, :3]))
    drot = math.degrees(math.acos(min(1.0, (np.trace(Wc[:3, :3] @ Wg[:3, :3].T) - 1) / 2)))
    dfoc = abs(cam.parm("focal").eval() - gt.parm("focal").eval()) / gt.parm("focal").eval() * 100
    return dpos, drot, dfoc


def pick_corners(pm, room, cam, gt, n=6):
    """Mesh points visible (inside frame, in front) in both cameras, spread out in the GT image."""
    P = np.array(room.displayNode().geometry().pointFloatAttribValues("P")).reshape(-1, 3)
    P = P @ pm._m(room.worldTransform())[:3, :3] + pm._m(room.worldTransform())[3, :3]
    geo = room.displayNode().geometry()           # room is not transformed in the test scene
    ok = np.ones(len(P), bool)
    uvs = []
    for c in (cam, gt):
        rig = pm.Rig(c)
        W = rig.world(rig.q)
        uv, z = rig.project(W, rig.q[6], P)
        ok &= (z > 0.5) & np.all((uv > 0.06) & (uv < 0.94), axis=1)
        for i in np.flatnonzero(ok):                # not occluded from this camera
            o = hou.Vector3(W[3, :3])
            d = hou.Vector3(P[i] - W[3, :3])
            hit = hou.Vector3()
            if geo.intersect(o, d.normalized(), hit, hou.Vector3(), hou.Vector3()) >= 0 and \
                    (hit - o).length() < d.length() - 1e-3:
                ok[i] = False
        uvs.append(uv)
    P, uv_gt = P[ok], uvs[1][ok]
    _, keep = np.unique(np.round(P, 4), axis=0, return_index=True)
    P, uv_gt = P[keep], uv_gt[keep]
    if n is None:
        return P, uv_gt
    chosen = [int(np.argmin(np.linalg.norm(uv_gt - 0.5, axis=1)))]
    while len(chosen) < n:
        d = np.min(np.linalg.norm(uv_gt[:, None] - uv_gt[chosen][None], axis=2), axis=1)
        chosen.append(int(np.argmax(d)))
    return P[chosen], uv_gt[chosen]


def next_corner(pm, room, cam, gt, used_uv):
    """A corner visible in the plate (GT camera) and inside the current frame, far (in the plate)
    from the used ones. It may be hidden in the current camera: the wireframe shows it anyway."""
    P, uv = pick_corners(pm, room, gt, gt, n=None)
    rig = pm.Rig(cam)
    uv_now, z = rig.project(rig.world(rig.q), rig.q[6], P)
    inside = (z > 0.2) & np.all((uv_now > 0.03) & (uv_now < 0.97), axis=1)
    P, uv = P[inside], uv[inside]
    if not used_uv:
        i = int(np.argmin(np.linalg.norm(uv - 0.5, axis=1)))
    else:
        i = int(np.argmax(np.min(np.linalg.norm(uv[:, None] - np.array(used_uv)[None], axis=2), axis=1)))
    return P[i], uv[i]


# ------------------------------------------------------------------ the test
def run():
    t0 = time.time()
    hou.hda.installFile(HDA)       # the scene records the library by absolute/$JOB path: not portable
    hou.hipFile.load(SCENE, suppress_save_prompt=True, ignore_load_warnings=True)
    hou.setFrame(1)
    sv = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
    node = hou.node("/obj/camera_pin_matcher1")
    pm = node.hdaModule()
    cam, gt, room = hou.node("/obj/cam1"), hou.node("/obj/gt_cam"), hou.node("/obj/room")
    room_sum = checksum(room)
    cam_start = [cam.parm(p).eval() for p in pm.PARMS]
    gt_sum = gt.asCode()
    node.setCurrent(True, clear_all_selected=True)
    sv.enterCurrentNodeState()
    refresh()
    st = live_state(node)
    drv = Driver(st)
    vp = st.vp
    ok = True

    def check(cond, what):
        nonlocal ok
        ok &= bool(cond)
        log(("PASS " if cond else "FAIL ") + what)

    # --- view lock
    check(vp.camera() == node.node("view_proxy") and not vp.isCameraLockedToView(),
          "viewport looks through the tool view (proxy of the target camera), unlinked")
    vp.setCamera(gt)                     # simulate the user switching cameras / tumbling away
    st._watchdog()
    check(vp.camera() == node.node("view_proxy"), "watchdog re-attaches the camera view")
    cam_before = [cam.parm(p).eval() for p in pm.PARMS]
    drv.wheel((400, 300), 3)
    zoom = node.parm("view_zoom").eval()
    check(zoom > 1.9 and [cam.parm(p).eval() for p in pm.PARMS] == cam_before,
          "mouse wheel zooms the 2D view (zoom %.2f) without touching the camera" % zoom)
    for k in ("space",):
        check(drv.key(k) is True, "'%s' is consumed (no volatile view tool)" % k)
    drv.menu("reset_view")
    check(node.parm("view_zoom").eval() == 1.0, "reset 2D view")

    # --- pins, one at a time: Ctrl+drag from a mesh corner onto its plate feature (create + align in
    #     one gesture, as a user would). Pin 1 must stay put while pin 2 is dragged.
    n_undo = len(hou.undos.undoLabels())
    worst_pin1, snap_err, steps, used = 0.0, 0.0, 12, []
    noise = np.random.default_rng(5).normal(0.0, 0.5, (6, 2))    # user placement accuracy, px
    jitter = np.array([3.0, -2.0])                              # click a bit off the corner: snapping
    log("| pins | status | position error | rotation error | focal error | RMS |")
    log("|---|---|---|---|---|---|")
    for i in range(6):
        p, target = next_corner(pm, room, cam, gt, used)
        used.append(target)
        fr = st._frame()
        a = st._screen(fr, fr.rig.project(fr.W, fr.f, p[None])[0])[0] + jitter
        b = st._screen(fr, [target])[0] + jitter + noise[i] * np.array([1, -1])   # grab offset kept
        drv._send(a[0], a[1], R.Start, left=True, ctrl=True)
        for k in range(1, steps + 1):
            t = k / float(steps)
            drv._send(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, R.Active, left=True, ctrl=True)
            if i == 1:
                f2 = st._frame()
                p1 = st.drag["data"]["frames"]["1"][0]
                uv1, _ = f2.rig.project(f2.W, f2.f, np.array([p1["p"]]))
                worst_pin1 = max(worst_pin1, f2.rig.pixel_errors(uv1, [p1["uv"]])[0])
        drv._send(b[0], b[1], R.Changed)
        refresh()
        snap_err = max(snap_err, np.linalg.norm(np.array(pm.pins_at(pm.load_pins(node))[-1]["p"]) - p))
        if i >= 2:
            dpos, drot, dfoc = cam_error(pm, cam, gt)
            h = st._hud_values()
            log("| %d | %s | %.4f | %.3f deg | %.3f %% | %s |" % (i + 1, h["status"], dpos, drot, dfoc, h["rms"]))
    pins = pm.pins_at(pm.load_pins(node))
    ids = [p["id"] for p in pins]
    check(len(pins) == 6 and snap_err < 1e-4, "6 pins created by Ctrl+drag, point snapping exact (max error %.1e)" % snap_err)
    check(worst_pin1 < 0.1, "aligned pin 1 stays in place while pin 2 is dragged (max drift %.4f px)" % worst_pin1)
    labels = hou.undos.undoLabels()[:len(hou.undos.undoLabels()) - n_undo]
    check(len(labels) == 6 and all("Create pin" in l for l in labels), "each create+drag is one undo step: %s" % (labels,))

    dpos, drot, dfoc = cam_error(pm, cam, gt)
    info = st._hud_values()
    log("after 6 drags: dpos %.4f  drot %.3f deg  dfocal %.3f%%  HUD status '%s' rms %s"
        % (dpos, drot, dfoc, info["status"], info["rms"]))
    drv.menu("solve_key")
    dpos, drot, dfoc = cam_error(pm, cam, gt)
    log("after Solve & Key: dpos %.4f  drot %.3f deg  dfocal %.3f%%" % (dpos, drot, dfoc))
    log("| parm | ground truth | solved | start |")
    log("|---|---|---|---|")
    start = dict(zip(pm.PARMS, cam_start))
    for p_ in pm.PARMS:
        log("| %s | %.4f | %.4f | %.4f |" % (p_, gt.parm(p_).evalAtFrame(1), cam.parm(p_).eval(), start[p_]))
    check(dfoc < 1.0 and dpos < 0.05 and drot < 0.3, "ground truth recovered (focal within 1%)")
    check(pm.keyed_frames(cam) == [1], "camera keyed at frame 1 only: %s" % pm.keyed_frames(cam))
    check(checksum(room) == room_sum, "reference mesh + network unchanged (checksum)")
    check(gt.asCode() == gt_sum, "other objects untouched")

    # --- undo of a drag restores pins and camera together
    before = [cam.parm(p).eval() for p in pm.PARMS], node.parm("pins").eval()
    fr = st._frame()
    pin = pm.pins_at(pm.load_pins(node))[2]
    a = st._screen(fr, [pin["uv"]])[0]
    drv.drag(a, a + np.array([40.0, 25.0]))
    moved = [cam.parm(p).eval() for p in pm.PARMS] != before[0]
    hou.undos.performUndo()
    after = [cam.parm(p).eval() for p in pm.PARMS], node.parm("pins").eval()
    check(moved and np.allclose(after[0], before[0]) and after[1] == before[1],
          "one undo reverts a whole drag (pins + camera)")

    # --- locks: focal locked -> never changes; nodal -> position never changes
    for preset, idx in (("preset_focal", [6]), ("preset_nodal", [0, 1, 2])):
        drv.menu("preset_clear")
        drv.menu(preset)
        vals = [cam.parm(p).eval() for p in pm.PARMS]
        fr = st._frame()
        pin = pm.pins_at(pm.load_pins(node))[3]
        a = st._screen(fr, [pin["uv"]])[0]
        drv.drag(a, a + np.array([-30.0, 20.0]))
        now = [cam.parm(p).eval() for p in pm.PARMS]
        check(all(now[i] == vals[i] for i in idx) and now != vals, "%s: locked parms unchanged during solve" % preset)
        hou.undos.performUndo()
    drv.menu("preset_clear")

    # --- a real click may arrive as press + release followed by 'Picked': still one pin
    n_before = len(pm.pins_at(pm.load_pins(node)))
    fr = st._frame()
    corner = pick_corners(pm, room, cam, cam, n=None)[0][-1]
    c = st._screen(fr, fr.rig.project(fr.W, fr.f, corner[None])[0])[0]
    drv._send(c[0], c[1], R.Start, left=True, ctrl=True)
    drv._send(c[0], c[1], R.Changed, ctrl=True)
    drv._send(c[0], c[1], R.Picked, left=True, ctrl=True)
    check(len(pm.pins_at(pm.load_pins(node))) == n_before + 1, "Ctrl+click as press/release/picked creates exactly one pin")
    hou.undos.performUndo()
    check(len(pm.pins_at(pm.load_pins(node))) == n_before, "pin creation is undoable")

    # --- pin flags, selection, delete, deactivate/unlock all
    st.selected = ids[5]
    drv.menu("toggle_active")
    drv.menu("toggle_lock")
    p6 = next(p for p in pm.pins_at(pm.load_pins(node)) if p["id"] == ids[5])
    check(not p6["on"] and p6["lock"], "toggle active / lock on the selected pin")
    fr = st._frame()
    a = st._screen(fr, [p6["uv"]])[0]
    drv.drag(a, a + np.array([50.0, 0.0]))
    p6b = next(p for p in pm.pins_at(pm.load_pins(node)) if p["id"] == ids[5])
    check(p6b["uv"] == p6["uv"], "locked pin can't be dragged")
    drv.menu("unlock_all")
    drv.menu("deactivate_all")
    ps = pm.pins_at(pm.load_pins(node))
    check(all(not p["lock"] for p in ps) and all(not p["on"] for p in ps), "unlock all / deactivate all")
    hou.undos.performUndo()
    hou.undos.performUndo()
    ps = pm.pins_at(pm.load_pins(node))
    check(ps[5]["lock"] and sum(p["on"] for p in ps) == 5, "unlock all / deactivate all are undoable")
    drv.menu("unlock_all")
    st.selected = ids[5]
    drv.menu("toggle_active")
    st.selected = ids[4]
    check(drv.key("Backspace") is True and len(pm.pins_at(pm.load_pins(node))) == 5, "Backspace deletes the selected pin")
    hou.undos.performUndo()
    check(len(pm.pins_at(pm.load_pins(node))) == 6, "pin delete is undoable")

    # --- timeline: frame 12, copy pins, ghosts, drag to GT, key, interpolation
    hou.setFrame(12)
    refresh()
    drv.menu("copy_nearest")
    d = pm.load_pins(node)
    check([p["id"] for p in pm.pins_at(d, 12)] == ids, "pins copied from nearest keyed frame (same ids)")
    check(pm.neighbour_frames(d, 12) == (1, None), "ghost source frames for frame 12: %s" % (pm.neighbour_frames(d, 12),))
    rig_gt = pm.Rig(gt)
    uv12 = rig_gt.project(rig_gt.world(rig_gt.q), rig_gt.q[6], np.array([p["p"] for p in pm.pins_at(d, 12)]))[0]
    for pid, target in zip(ids, uv12):
        fr = st._frame()
        pin = next(p for p in pm.pins_at(pm.load_pins(node), 12) if p["id"] == pid)
        drv.drag(st._screen(fr, [pin["uv"]])[0], st._screen(fr, [target])[0])
    drv.menu("solve_key")
    dpos, drot, dfoc = cam_error(pm, cam, gt)
    log("frame 12 after copy + 6 drags + Solve & Key: dpos %.4f  drot %.3f deg  dfocal %.3f%%" % (dpos, drot, dfoc))
    check(dfoc < 1.0 and dpos < 0.05, "frame 12 solved from copied pins")
    check(pm.keyed_frames(cam) == [1, 12], "keys on frames 1 and 12: %s" % pm.keyed_frames(cam))
    k = cam.parm("tx").keyframes()
    check(all(kf.expression() == "bezier()" for kf in k) and cam.parm("tx").evalAtFrame(6) != cam.parm("tx").evalAtFrame(1),
          "keys use bezier() and the camera interpolates between keys")
    hou.setFrame(24)
    refresh()
    check(pm.neighbour_frames(pm.load_pins(node), 24) == (12, None), "ghosts on frame 24 come from frame 12")
    hou.setFrame(12)
    with hou.undos.group("test: delete keys"):
        pm.delete_keys(cam)
    check(pm.keyed_frames(cam) == [1], "delete keys on current frame")
    hou.undos.performUndo()
    check(pm.keyed_frames(cam) == [1, 12], "delete keys is undoable")
    n12 = pm.delete_frame_pins(node, 12)
    check(n12 == 6 and not pm.pins_at(pm.load_pins(node), 12), "delete pins on current frame")
    hou.undos.performUndo()
    check(len(pm.pins_at(pm.load_pins(node), 12)) == 6, "per-frame delete is undoable")
    tot = pm.delete_all_pins(node, confirm=False)
    check(tot == 12 and not pm.pin_frames(pm.load_pins(node)), "delete all pins")
    hou.undos.performUndo()
    check(pm.pin_frames(pm.load_pins(node)) == [1, 12], "delete all is undoable")

    # --- display modes (visual check images)
    hou.setFrame(1)
    for mode in ("bg", "fg"):
        node.parm("platemode").set(mode)
        refresh(st)
        img = os.path.join(ROOT, "tests", "gui_%s.png" % mode)
        hou.qt.mainWindow().grab().save(img)
        log("saved screenshot", os.path.relpath(img, ROOT))
    node.parm("platemode").set("bg")

    # --- exit state restores the viewport, persistence across save/load
    sv.setCurrentState("objview")
    check(vp.camera() != node.node("view_proxy") and vp.settings().visibleObjects() == "*",
          "leaving the tool restores the viewport (camera, visible objects)")
    check(checksum(room) == room_sum, "reference mesh still unchanged after all operations")
    pins_json, keys = node.parm("pins").eval(), {p: [(k.frame(), k.value()) for k in cam.parm(p).keyframes()] for p in pm.PARMS}
    tmp = os.path.join(tempfile.mkdtemp(), "persist.hip" + EXT)
    hou.hipFile.save(tmp)
    hou.hipFile.clear(suppress_save_prompt=True)
    hou.hipFile.load(tmp, suppress_save_prompt=True, ignore_load_warnings=True)
    n2, c2 = hou.node("/obj/camera_pin_matcher1"), hou.node("/obj/cam1")
    keys2 = {p: [(k.frame(), k.value()) for k in c2.parm(p).keyframes()] for p in pm.PARMS}
    check(n2.parm("pins").eval() == pins_json and keys2 == keys, "pins and keys survive save + reload")
    log("elapsed %.1fs" % (time.time() - t0))
    log("ALL PASSED" if ok else "SOME CHECKS FAILED")
    return ok


def main():
    try:
        ok = run()
    except Exception:
        log(traceback.format_exc())
        ok = False
    with open(LOG, "w") as f:
        f.write("\n".join(lines) + "\n")
    return ok


if __name__ == "__rpc__":                   # hython dev/rpc.py tests/test_gui.py (open dev session)
    main()
elif __name__ in ("__main__", "__builtin__", "builtins") and hou.isUIAvailable() and "--no-exit" not in sys.argv:
    hdefereval.executeDeferred(lambda: (main(), hou.exit(suppress_save_prompt=True)))
