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
    q, info = pm.solve(rig, P, uv, [True] * 7, rig.q, lock_roll=True)
    roll0, roll1 = pm._roll(rig.world(rig.q)[:3, :3]), pm._roll(rig.world(q)[:3, :3])
    assert abs(roll1 - roll0) < 1e-6, (roll0, roll1)
    q, info = pm.solve(rig, P, uv * 0 + 0.5, [True] * 7, rig.q, focal_range=(30.0, 40.0))
    assert 30.0 - 1e-9 <= q[6] <= 40.0 + 1e-9
    print("locks: locked values bit-identical, roll lock holds (%.1e rad), focal clamped to %.3f" % (abs(roll1 - roll0), q[6]))


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and (len(sys.argv) < 2 or name in sys.argv[1:]):
            fn()
    print("OK")
