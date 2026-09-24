"""Core checks for the Camera Pin Matcher module (camera model + solver).

    hython tests/test_core.py
"""
import math
import os
import sys
import time

import numpy as np

import hou

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import pinmatch as pm  # noqa: E402

rng = np.random.default_rng(7)


def make_cam(name="cam", parent=None, **parms):
    cam = hou.node("/obj").createNode("cam", name)
    if parent is not None:
        cam.setFirstInput(parent)
    for k, v in parms.items():
        (cam.parmTuple(k) if isinstance(v, tuple) else cam.parm(k)).set(v)
    return cam


def houdini_ndc(cam, P):
    """Ground truth: VEX toNDC() on the camera."""
    g = hou.Geometry()
    g.createPoints([tuple(p) for p in P])
    obj = hou.node("/obj").createNode("geo")
    stash = obj.createNode("stash")
    stash.parm("stash").set(g)
    w = obj.createNode("attribwrangle")
    w.setInput(0, stash)
    w.parm("snippet").set('v@ndc = toNDC("%s", @P);' % cam.path())
    ndc = np.array(w.geometry().pointFloatAttribValues("ndc")).reshape(-1, 3)[:, :2]
    obj.destroy()
    return ndc


def test_camera_model():
    worst_w = worst_uv = 0.0
    for i in range(12):
        par = hou.node("/obj").createNode("null")
        par.parmTuple("t").set(tuple(rng.uniform(-3, 3, 3)))
        par.parmTuple("r").set(tuple(rng.uniform(-40, 40, 3)))
        par.parmTuple("s").set(tuple(rng.uniform(0.5, 2.0, 3)) if i % 2 else (1.7, 1.7, 1.7))
        cam = make_cam("c%d" % i, par, t=tuple(rng.uniform(-2, 2, 3)), r=tuple(rng.uniform(-30, 30, 3)),
                       p=tuple(rng.uniform(-0.5, 0.5, 3)), pr=tuple(rng.uniform(-10, 10, 3)),
                       s=tuple(rng.uniform(0.5, 2, 3)), scale=float(rng.uniform(0.5, 2)),
                       focal=float(rng.uniform(15, 120)), aperture=float(rng.uniform(20, 40)),
                       res=(int(rng.integers(320, 4000)), int(rng.integers(240, 3000))),
                       aspect=float(rng.choice([1.0, 1.333, 0.9])),
                       winx=float(rng.uniform(-0.1, 0.1)), winy=float(rng.uniform(-0.1, 0.1)),
                       winsizex=float(rng.uniform(0.7, 1.2)), winsizey=float(rng.uniform(0.7, 1.2)))
        cam.parm("xOrd").set(["srt", "str", "rst", "rts", "tsr", "trs"][i % 6])
        cam.parm("rOrd").set(["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"][(i * 5) % 6])
        if i % 3 == 0:
            cam.setPreTransform(hou.hmath.buildTransform({"translate": (0.2, -0.1, 0.3), "rotate": (5, -7, 3)}))
        rig = pm.Rig(cam)
        W = rig.world(rig.q)
        worst_w = max(worst_w, np.abs(W - pm._m(cam.worldTransform())).max())
        Wi = np.linalg.inv(W)
        P = (np.c_[rng.uniform(-1, 1, (25, 2)), -rng.uniform(1, 20, 25)]) @ W[:3, :3] + W[3, :3]
        uv, z = rig.project(W, rig.q[6], P)
        worst_uv = max(worst_uv, np.abs(uv - houdini_ndc(cam, P)).max())
        back = rig.unproject(W, rig.q[6], uv, 3.0)
        uv2, z2 = rig.project(W, rig.q[6], back)
        assert np.allclose(uv2, uv) and np.allclose(z2, 3.0)
    print("camera model: max |W - worldTransform| = %.2e, max |uv - toNDC| = %.2e" % (worst_w, worst_uv))
    assert worst_w < 1e-5 and worst_uv < 1e-5


def scene_points(n):
    """Box-room-like points spread in depth, in front of a camera at the origin looking -Z."""
    return np.c_[rng.uniform(-3, 3, n), rng.uniform(-1.5, 2, n), -rng.uniform(3, 12, n)]


def perturbed(q, dt=0.4, dr=4.0, focal=None):
    q = np.array(q, float)
    q[:3] += rng.uniform(-dt, dt, 3)
    q[3:6] += rng.uniform(-dr, dr, 3)
    if focal:
        q[6] = focal
    return q


def test_ground_truth():
    cam = make_cam("gt", t=(0.5, 1.6, 2.0), r=(-6.0, 10.0, 1.0), focal=32.0, aperture=36.0, res=(1920, 1080))
    rig = pm.Rig(cam)
    gt = rig.q.copy()
    Wg = rig.world(gt)
    for n, noise in ((6, 0.0), (6, 0.5), (12, 0.5), (30, 0.5)):
        P = scene_points(n) @ Wg[:3, :3] + Wg[3, :3]
        uv, _ = rig.project(Wg, gt[6], P)
        uv = uv + rng.normal(0, noise, uv.shape) / rig.res
        start = perturbed(gt, focal=45.0)
        t0 = time.time()
        q, info = pm.solve(rig, P, uv, [True] * 7, start, polish=2)
        dt = time.time() - t0
        W = rig.world(q)
        dpos = np.linalg.norm(W[3, :3] - Wg[3, :3])
        dang = math.degrees(math.acos(min(1.0, (np.trace(W[:3, :3] @ Wg[:3, :3].T) - 1) / 2)))
        dfoc = abs(q[6] - gt[6]) / gt[6] * 100
        print("GT %2d pins noise %.1fpx: dpos %.4f  drot %.3f deg  dfocal %.3f%%  rms %.3fpx  %s  %d it  %.1f ms"
              % (n, noise, dpos, dang, dfoc, info["rms"], info["status"], info["iterations"], dt * 1000))
        assert info["status"] == "over-constrained"
        assert dfoc < (0.05 if noise == 0 else 1.0) and dpos < (0.005 if noise == 0 else 0.1)
        if n == 30:
            assert dt < 0.1, "too slow for live dragging"


def test_locks():
    cam = make_cam("lk", t=(0.3, 1.5, 2.5), r=(-5.0, 8.0, 0.0), focal=35.0, res=(1280, 720))
    rig = pm.Rig(cam)
    W = rig.world(rig.q)
    P = scene_points(8) @ W[:3, :3] + W[3, :3]
    uv, _ = rig.project(W, rig.q[6], P)
    uv += rng.uniform(-0.03, 0.03, uv.shape)                 # inconsistent targets: pulls everything
    for locked in ([6], [0, 1, 2], [3, 4, 5], [0, 4, 6], [5]):
        free = [i not in locked for i in range(7)]
        q, info = pm.solve(rig, P, uv, free, rig.q)
        assert all(q[i] == rig.q[i] for i in locked), (locked, q, rig.q)
        assert any(abs(q[i] - rig.q[i]) > 1e-6 for i in range(7) if free[i])
    roll = lambda r, q: pm._roll(r.world(q)[:3, :3])
    worst = 0.0
    for locked in ([], [6], [0, 1, 2], [3], [4, 6]):              # exact with any mix of parm locks
        q, info = pm.solve(rig, P, uv, [i not in locked for i in range(7)], rig.q, lock_roll=True, polish=2)
        worst = max(worst, abs(roll(rig, q) - roll(rig, rig.q)))
        assert info["roll_off"] is None and np.any(np.abs(q[3:6] - rig.q[3:6]) > 1e-3), (locked, q)
    assert worst < 1e-12, worst
    # Roll is undefined looking straight down: the lock is off within VERTICAL_DEG (and says so), exact outside.
    for tilt, off in ((-88.0, True), (-80.0, False)):
        rv = pm.Rig(make_cam("lv%d" % -tilt, t=(0.0, 12.0, 0.0), r=(tilt, 20.0, 0.0), focal=35.0, res=(1280, 720)))
        Wv = rv.world(rv.q)
        Pv = scene_points(8) @ Wv[:3, :3] + Wv[3, :3]
        uvv = rv.project(Wv, rv.q[6], Pv)[0] + rng.uniform(-0.02, 0.02, (8, 2))
        q, info = pm.solve(rv, Pv, uvv, [True] * 7, rv.q, lock_roll=True, polish=2)
        assert (info["roll_off"] is not None) == off and (off or abs(roll(rv, q) - roll(rv, rv.q)) < 1e-12), (tilt, info)
    q, info = pm.solve(rig, P, uv * 0 + 0.5, [True] * 7, rig.q, focal_range=(30.0, 40.0))
    assert 30.0 - 1e-9 <= q[6] <= 40.0 + 1e-9
    print("locks: locked values bit-identical, roll lock exact (%.1e rad) and off within %g deg of vertical, "
          "focal clamped to %.3f" % (worst, pm.VERTICAL_DEG, q[6]))


def test_outlier():
    """6 good pins and 1 on the wrong corner: the robust solve keeps the camera and flags that pin."""
    r = np.random.default_rng(1)                         # own generator: same data when run alone
    cam = make_cam("ol", t=(0.5, 1.6, 2.0), r=(-6.0, 10.0, 1.0), focal=32.0, aperture=36.0, res=(1920, 1080))
    rig = pm.Rig(cam)
    gt = rig.q.copy()
    Wg = rig.world(gt)
    P = np.c_[r.uniform(-3, 3, 7), r.uniform(-1.5, 2, 7), -r.uniform(3, 12, 7)] @ Wg[:3, :3] + Wg[3, :3]
    uv = rig.project(Wg, gt[6], P)[0] + r.normal(0, 0.5, (7, 2)) / rig.res
    uv[6] += np.array([70.0, -50.0]) / rig.res          # a neighbouring corner of the plate
    start = gt + np.r_[r.uniform(-0.4, 0.4, 3), r.uniform(-4, 4, 3), 13.0]    # focal 45
    res = {}
    for robust in (False, True):
        q, info = pm.solve(rig, P, uv, [True] * 7, start, polish=2, robust=robust)
        W = rig.world(q)
        res[robust] = (np.linalg.norm(W[3, :3] - Wg[3, :3]), abs(q[6] - gt[6]) / gt[6] * 100, info)
    (dpos, dfoc, info), (dpos0, dfoc0, info0) = res[True], res[False]
    qg, _ = pm.solve(rig, P[:6], uv[:6], [True] * 7, start, polish=2, robust=False)    # the good pins alone
    qb, ib = pm.solve(rig, P[:6], uv[:6], [True] * 7, start, polish=2)
    Wg6 = rig.world(qg)
    print("outlier: least squares dpos %.4f dfocal %.2f%% flags %s | robust dpos %.4f dfocal %.3f%% flags %s | "
          "6 good pins alone dpos %.4f dfocal %.3f%%" % (
              dpos0, dfoc0, np.flatnonzero(info0["outliers"]), dpos, dfoc, np.flatnonzero(info["outliers"]),
              np.linalg.norm(Wg6[3, :3] - Wg[3, :3]), abs(qg[6] - gt[6]) / gt[6] * 100))
    assert dfoc0 > 2.0, "least squares should be dragged off, or this test proves nothing"
    q = pm.solve(rig, P, uv, [True] * 7, start, polish=2)[0]
    assert np.linalg.norm(rig.world(q)[3, :3] - Wg6[3, :3]) < 0.01 and abs(q[6] - qg[6]) / qg[6] < 0.002, \
        "the wrong pin has next to no pull: as good as leaving it out"
    assert dfoc < 2.0 and dpos < 0.1 and list(np.flatnonzero(info["outliers"])) == [6]
    assert np.array_equal(qg, qb) and not ib["outliers"].any(), "pins that agree: robust = least squares"


def test_few_pins():
    cam = make_cam("fp", t=(0.0, 1.6, 3.0), r=(-5.0, 0.0, 0.0), focal=30.0, res=(1920, 1080))
    rig = pm.Rig(cam)
    q0 = rig.q.copy()
    W0 = rig.world(q0)
    P = np.array([[-1.0, 0.5, -3.0], [1.5, 1.2, -6.0]]) @ W0[:3, :3] + W0[3, :3]
    uv0, _ = rig.project(W0, q0[6], P)

    # 1 pin dragged 150 px: camera should only pan/tilt.
    tgt = uv0[:1] + np.array([150.0, -80.0]) / rig.res
    q1, info = pm.solve(rig, P[:1], tgt, [True] * 7, q0)
    W1 = rig.world(q1)
    move = np.linalg.norm(W1[3, :3] - W0[3, :3])
    print("1 pin : err %.4f px, camera moved %.5f, focal %.3f -> %.3f, rot change %s  [%s]"
          % (info["rms"], move, q0[6], q1[6], np.round(q1[3:6] - q0[3:6], 3), info["status"]))
    assert info["rms"] < 0.05 and move < 1e-3 and abs(q1[6] - q0[6]) < 0.05 and info["status"] == "under-constrained"

    # 2 pins: pin A stays, pin B dragged. Focal free -> zoom/roll, focal locked -> dolly.
    tgt = np.array([uv0[0], uv0[1] + np.array([90.0, 60.0]) / rig.res])
    for free_focal in (True, False):
        q2, info = pm.solve(rig, P, tgt, [True] * 6 + [free_focal], q0)
        W2 = rig.world(q2)
        uv2, _ = rig.project(W2, q2[6], P)
        errA = rig.pixel_errors(uv2[:1], tgt[:1])[0]
        d = (W2[3, :3] - W0[3, :3]) @ W0[:3, :3].T
        print("2 pins focal %s: pinA err %.4f px, pinB err %.4f px, focal %.2f, move (x,y,z) %s  [%s]"
              % ("free  " if free_focal else "locked", errA, info["errors"][1], q2[6], np.round(d, 4), info["status"]))
        assert errA < 0.05 and info["errors"][1] < 0.05
        if free_focal:
            assert abs(q2[6] - q0[6]) > 0.2 and np.linalg.norm(d) < 0.045   # < 1% of pin depth
        else:
            assert abs(d[2]) > 5 * max(abs(d[0]), abs(d[1]))   # mostly dolly


def test_cheirality():
    cam = make_cam("ch", t=(0.0, 0.0, 0.0), r=(0.0, 0.0, 0.0), focal=30.0, res=(1000, 1000))
    rig = pm.Rig(cam)
    P = np.array([[0.0, 0.0, -2.0], [0.5, 0.0, -2.5], [0.0, 0.5, -3.0], [-0.4, -0.3, -2.2]])
    tgt = np.array([[1.8, 0.5], [-0.9, 0.5], [0.5, 1.9], [0.5, -0.8]])   # absurd targets
    q, info = pm.solve(rig, P, tgt, [True] * 7, rig.q, focal_range=(5.0, 500.0))
    _, z = rig.project(rig.world(q), q[6], P)
    print("cheirality: min depth %.3f, focal %.2f, status %s" % (z.min(), q[6], info["status"]))
    assert (z > 0).all() and 5.0 - 1e-9 <= q[6] <= 500.0 + 1e-9 and np.all(np.isfinite(q))


def test_protected_parms():
    cam = make_cam("pp", t=(0.0, 1.0, 5.0), focal=35.0, res=(1280, 720))
    cam.parm("tx").setExpression("sin($F) * 0.1")                 # expression -> protected
    cam.parm("ry").lock(True)                                     # padlock -> protected
    for f, v in ((1, 5.0), (10, 4.0)):                            # plain animation is fine
        cam.parm("tz").setKeyframe(hou.Keyframe(v, hou.frameToTime(f)))
    fatal, prot = pm.camera_problems(cam)
    assert fatal is None and prot == {"tx": "expression/reference", "ry": "locked"}, prot
    before = [cam.parm(n).eval() for n in pm.PARMS]
    free = [n not in prot for n in pm.PARMS]
    with hou.undos.disabler():
        pm.write_camera(cam, np.array(before) + 1.0, free)
    after = [cam.parm(n).eval() for n in pm.PARMS]
    assert after[0] == before[0] and after[4] == before[4] and after[1] == before[1] + 1.0
    cam.parm("lookatpath").set("/obj")
    assert pm.camera_problems(cam)[0] == "camera uses Look At"
    print("protected parms: %s treated as locked, look-at disables solving" % sorted(prot))


def test_splat_hit():
    """Ray hits on Gaussian splats and plain point clouds: the first visible surface, not stray points."""
    q = rng.normal(size=(20, 4))
    R = pm._quat_matrices(q)
    for qi, Ri in zip(q, R):                       # local = world @ R, i.e. R is the transpose of Houdini's
        hq = hou.Quaternion(*(qi / np.linalg.norm(qi)))
        assert np.allclose(Ri, np.array(hq.extractRotationMatrix3().asTupleOfTuples()).T, atol=1e-6)
    # a wall of flat splats facing the camera at z = -5 (another at z = -8), faint floaters in front
    n = 40000
    wall = np.c_[rng.uniform(-3, 3, (n, 2)), np.full(n, -5.0)]
    back = wall + [0.0, 0.0, -3.0]
    floaters = np.c_[rng.uniform(-3, 3, (300, 2)), rng.uniform(-4.5, -0.5, 300)]
    P = np.r_[wall, back, floaters]
    spin = np.c_[np.zeros((2 * n, 2)), np.sin(rng.uniform(0, np.pi, 2 * n)), np.ones(2 * n)]   # about z
    orient = np.r_[spin, rng.normal(size=(300, 4))]
    sigma = np.r_[np.tile([0.05, 0.03, 0.002], (2 * n, 1)), np.full((300, 3), 0.05)]
    alpha = np.r_[np.full(2 * n, 0.8), np.full(300, 0.2)]
    worst = 0.0
    for d in rng.uniform(-0.4, 0.4, (30, 2)):
        d = np.r_[d, -1.0]
        t, i = pm.splat_hit((0, 0, 0), d, P, sigma, alpha, orient)
        worst = max(worst, abs(t * d[2] + 5.0))
        assert i < n, "strongest splat must be on the front wall"
    assert worst < 0.01, worst
    t, i = pm.splat_hit((0, 0, 0), P[2 * n] * [1, 1, 1], P, sigma, alpha * 0 + 0.99, orient)
    assert i == 2 * n and abs(t - 1.0) < 1e-6, "an opaque floater on the ray is what the ray sees"
    # plain point cloud: pixel-sized Gaussians, same scene without splat attributes
    worst_pc = 0.0
    for d in rng.uniform(-0.4, 0.4, (30, 2)):
        d = np.r_[d, -1.0]
        t, i = pm.splat_hit((0, 0, 0), d, P, px=1.0 / 1500)
        worst_pc = max(worst_pc, abs(t * d[2] + 5.0))
    assert worst_pc < 0.01, worst_pc
    assert pm.splat_hit((0, 0, 0), (0, 0, 1), P, sigma, alpha, orient) is None
    t0 = time.time()
    big = np.tile(P, (30, 1))                      # ~2.4 M splats
    pm.splat_hit((0, 0, 0), (0.1, 0.1, -1), big, np.tile(sigma, (30, 1)), np.tile(alpha, 30), np.tile(orient, (30, 1)))
    dt = time.time() - t0
    print("splats: depth error %.4f (splats) / %.4f (plain points), %d splats picked in %.0f ms" % (worst, worst_pc, len(big), dt * 1000))
    assert dt < 1.0


def test_polygons():
    g = hou.Geometry()
    hou.sopNodeTypeCategory().nodeVerb("box").execute(g, [])
    assert pm.polygons(g) is g
    soup, packed = hou.Geometry(), hou.Geometry()
    hou.sopNodeTypeCategory().nodeVerb("polysoup").execute(soup, [g])
    hou.sopNodeTypeCategory().nodeVerb("pack").execute(packed, [soup])
    for src in (soup, packed):
        out = pm.polygons(src)
        assert out.countPrimType(hou.primType.Polygon) == 6 == out.intrinsicValue("primitivecount"), src
    print("polygons: packed and polysoup references are converted for the drawables")


def test_hda():
    """Asset-level behaviour in a fresh scene: callbacks, resolution matching, undo grouping."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ext = {hou.licenseCategoryType.Commercial: "", hou.licenseCategoryType.Indie: "lc"}.get(hou.licenseCategory(), "nc")
    hou.hda.installFile(os.path.join(root, "otls", "camera_pin_matcher.hda" + ext))
    cam = make_cam("hcam", t=(0.4, 1.5, 3.0), r=(-5.0, 10.0, 0.0), focal=35.0, res=(640, 480))
    node = hou.node("/obj").createNode("pinmatch::camera_pin_matcher::1.0")
    node.parm("camera").set(cam.path())
    node.parm("plate").set(os.path.join(root, "scenes", "plate", "plate.$F4.jpg"))
    npm = node.hdaModule()
    node.parm("matchres").set(1)
    assert npm.match_resolution(node, cam) and cam.parmTuple("res").eval() == (1280, 720)
    rig = npm.Rig(cam)
    W = rig.world(rig.q)
    P = scene_points(6) @ W[:3, :3] + W[3, :3]
    uv, _ = rig.project(W, rig.q[6], P)
    data = npm.load_pins(node)
    for p, t in zip(P, uv + 0.01):                          # pins that ask for a small camera move
        npm.new_pin(data, 1, p, t)
    npm.save_pins(node, data)
    hou.setFrame(1)
    n_undo = len(hou.undos.undoLabels())
    node.parm("solvekey").pressButton()
    assert len(hou.undos.undoLabels()) == n_undo + 1, hou.undos.undoLabels()[:3]
    assert npm.keyed_frames(cam) == [1] and abs(cam.parm("focal").eval() - 35.0) > 1e-6
    hou.undos.performUndo()
    assert npm.keyed_frames(cam) == [] and cam.parm("focal").eval() == 35.0
    node.parm("preset_nodal").pressButton()
    assert [node.parm("lock_" + n).eval() for n in ("tx", "ty", "tz", "rx")] == [1, 1, 1, 0]
    hou.setFrame(5)
    node.parm("copynearest").pressButton()
    assert [p["id"] for p in npm.pins_at(npm.load_pins(node), 5)] == [1, 2, 3, 4, 5, 6]
    node.parm("deactivateall").pressButton()
    assert not any(p["on"] for f in npm.load_pins(node)["frames"].values() for p in f)
    node.parm("deleteframe").pressButton()
    assert npm.pin_frames(npm.load_pins(node)) == [1]
    print("hda: resolution matched, Solve & Key = 1 undo step, presets / copy / deactivate / delete buttons ok")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and (len(sys.argv) < 2 or name in sys.argv[1:]):
            fn()
    print("OK")
