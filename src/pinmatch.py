"""Camera Pin Matcher - HDA PythonModule.

Camera model, Levenberg-Marquardt solver, pin storage, camera read/write/keys and the
parameter-button callbacks. The viewer state (ViewerStateModule) uses this through
node.hdaModule(). Only the target camera is ever written; the reference geometry is only read.
"""
import json
import math
import re

import numpy as np

import hou

PARMS = ("tx", "ty", "tz", "rx", "ry", "rz", "focal")
UNDO_PREFIX = "Camera Pin Matcher: "
_CURVE_FUNCS = re.compile(r"^(bezier|linear|constant|cubic|qlinear|quintic|spline|ease|easein|easeout|"
                          r"easep|easeinp|easeoutp|match|matchin|matchout|vmatch|vmatchin|vmatchout)\(\)$")


def _m(hmat):
    return np.array(hmat.asTuple(), dtype=float).reshape(4, 4)


# --------------------------------------------------------------------------------------------
# Camera model
# --------------------------------------------------------------------------------------------
class Rig(object):
    """Snapshot of a camera: maps the 7 solve values q = (tx, ty, tz, rx, ry, rz, focal) to
    image space exactly like Houdini does:

        M = buildTransform(t, r, s * scale, p, pr; xOrd, rOrd) * preTransform * parentAndSubnet
        W = M with its 3x3 part orthonormalised (polar decomposition) - Houdini removes
            parent/own scale from camera transforms
        u = ((f/A) * Xc/-Zc - winx) / winsizex + 0.5,   v likewise with the vertical aperture
    Row-vector convention (as hou.Matrix4): p_world = p_cam * W.
    """

    def __init__(self, cam):
        pt = lambda n: tuple(cam.parmTuple(n).eval())
        self.q = np.array([cam.parm(n).eval() for n in PARMS], dtype=float)
        s = cam.parm("scale").eval()
        self.fixed = {"scale": tuple(v * s for v in pt("s")), "pivot": pt("p"), "pivot_rotate": pt("pr")}
        self.xord = cam.parm("xOrd").evalAsString()
        self.rord = cam.parm("rOrd").evalAsString()
        self.post = _m(cam.preTransform()) @ _m(cam.parentAndSubnetTransform())
        self.aperture = cam.parm("aperture").eval()
        self.res = np.array(pt("res"), dtype=float)
        self.pixel_aspect = cam.parm("aspect").eval()
        self.win = tuple(cam.parm(n).eval() for n in ("winx", "winy", "winsizex", "winsizey"))

    def world(self, q):
        xf = dict(self.fixed, translate=tuple(map(float, q[:3])), rotate=tuple(map(float, q[3:6])))
        m = _m(hou.hmath.buildTransform(xf, self.xord, self.rord)) @ self.post
        u, _, vt = np.linalg.svd(m[:3, :3])
        m[:3, :3] = u @ vt
        return m

    def scales(self, focal):
        fx = focal / self.aperture
        return fx, fx * self.res[0] * self.pixel_aspect / self.res[1]

    def project(self, W, focal, P):
        """World points (n,3) -> NDC uv (n,2), depth in front of the camera (n,)."""
        wi = np.linalg.inv(W)
        pc = np.asarray(P, float) @ wi[:3, :3] + wi[3, :3]
        z = -pc[:, 2]
        zs = np.where(np.abs(z) < 1e-12, 1e-12, z)
        fx, fy = self.scales(focal)
        wx, wy, sx, sy = self.win
        return np.stack([(fx * pc[:, 0] / zs - wx) / sx + 0.5, (fy * pc[:, 1] / zs - wy) / sy + 0.5], 1), z

    def unproject(self, W, focal, uv, depth):
        """NDC uv (n,2) -> world points on the camera rays at the given depth."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        fx, fy = self.scales(focal)
        wx, wy, sx, sy = self.win
        pc = np.stack([((uv[:, 0] - 0.5) * sx + wx) / fx * depth,
                       ((uv[:, 1] - 0.5) * sy + wy) / fy * depth,
                       -np.full(len(uv), float(depth))], 1)
        return pc @ W[:3, :3] + W[3, :3]

    def pixel_errors(self, uv, target):
        return np.hypot(*((np.asarray(uv) - np.asarray(target)) * self.res).T)


def _roll(C):
    """Angle of the camera X axis around the view axis relative to the world horizon (Y up)."""
    return math.atan2(C[0, 1], C[1, 1])


# --------------------------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------------------------
def solve(rig, P, UV, free, anchor, start=None, weight=1.0, lock_roll=False,
          focal_range=(1.0, 5000.0), iters=40, polish=0):
    """Levenberg-Marquardt over the free subset of q (focal solved as log f).

    Minimises pixel reprojection error of the pins plus a small "minimal camera change" prior
    toward `anchor` (pixel-equivalent units, see PLAN.md). Pins behind the anchor camera are
    ignored; steps that push a used pin behind the camera are rejected.
    `polish` re-anchors the prior at the result and solves again (removes the prior's bias in
    directions the pins determine; used on commit, not while dragging).
    Returns (q, info) - q is `start`/`anchor` unchanged if nothing could be solved.
    """
    P = np.asarray(P, float).reshape(-1, 3)
    UV = np.asarray(UV, float).reshape(-1, 2)
    anchor = np.asarray(anchor, float)
    base = np.array(anchor if start is None else start, float)
    free = np.asarray(free, bool)
    idx = np.flatnonzero(free)
    lock_roll = bool(lock_roll) and bool(free[3:6].any())
    dof = len(idx) - int(lock_roll)

    Wa = rig.world(anchor)
    Ca, ca = Wa[:3, :3].copy(), Wa[3, :3].copy()
    _, za = rig.project(Wa, anchor[6], P)
    use = za > 1e-9
    info = {"n": int(use.sum()), "behind": int((~use).sum()), "dof": dof, "iterations": 0}
    if not use.any() or not len(idx):
        info.update(status="no active pins" if not len(P) else ("all parameters locked" if not len(idx)
                    else "pins behind camera"), rms=0.0, errors=np.zeros(len(P)), rank=0)
        return base, info
    Pu, UVu = P[use], UV[use]

    F = anchor[6] / rig.aperture * rig.res[0]           # focal length in pixels
    R = 0.5 * math.hypot(*rig.res)                      # half image diagonal in pixels
    D = float(np.median(za[use]))                        # typical pin depth
    lam = 2e-4 * weight
    w_rot = lam * np.array([F, F, 8 * R])               # tilt, pan, roll
    w_zoom = lam * 8 * R
    w_move = lam * np.array([120 * F, 120 * F, 40 * R]) / D   # truck, pedestal, dolly
    roll_a = _roll(Ca)
    lo, hi = math.log(focal_range[0]), math.log(focal_range[1])

    def to_q(x):
        q = base.copy()
        q[idx] = x
        if free[6]:
            q[6] = math.exp(x[-1])
        return q

    def clip(x):
        if free[6]:
            x[-1] = min(max(x[-1], lo), hi)
        return x

    def residual(x):
        q = to_q(x)
        W = rig.world(q)
        uv, z = rig.project(W, q[6], Pu)
        C = W[:3, :3]
        Q = C @ Ca.T                                     # rotation relative to the anchor camera
        w = 0.5 * np.array([Q[1, 2] - Q[2, 1], Q[2, 0] - Q[0, 2], Q[0, 1] - Q[1, 0]])
        d = (W[3, :3] - ca) @ Ca.T                       # translation in anchor camera axes
        parts = [((uv - UVu) * rig.res).ravel(), w_rot * w, [w_zoom * math.log(q[6] / anchor[6])], w_move * d]
        if lock_roll:
            parts.append([1e3 * R * math.remainder(_roll(C) - roll_a, 2 * math.pi)])
        return np.concatenate(parts), z

    x = base[idx].copy()
    if free[6]:
        x[-1] = math.log(base[6])
    x = clip(x)
    r, z = residual(x)
    if not np.all(np.isfinite(r)):
        info.update(status="invalid camera", rms=0.0, errors=np.zeros(len(P)), rank=0)
        return base, info
    front = z > 1e-9
    zmin = 1e-6 * D
    steps = np.array([1e-6 * D] * 3 + [1e-5] * 3 + [1e-7])[idx]
    cost, mu = r @ r, 1e-3
    for it in range(iters):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            xj = x.copy()
            xj[j] += steps[j]
            J[:, j] = (residual(xj)[0] - r) / steps[j]
        A, g = J.T @ J, J.T @ r
        damp = np.diag(A) + 1e-12
        while mu < 1e12:
            dx = -np.linalg.solve(A + mu * np.diag(damp), g)
            xn = clip(x + dx)
            rn, zn = residual(xn)
            cn = rn @ rn
            if np.all(np.isfinite(rn)) and not np.any(front & (zn <= zmin)) and cn < cost:
                break
            mu *= 4.0
        else:
            break
        info["iterations"] = it + 1
        done = cost - cn < 1e-12 * (1.0 + cost) or np.max(np.abs(xn - x)) < 1e-12
        x, r, z, cost, mu = xn, rn, zn, cn, max(mu / 3.0, 1e-9)
        if done:
            break

    q = to_q(x)
    uv, _ = rig.project(rig.world(q), q[6], P)
    err = rig.pixel_errors(uv, UV)
    err[~use] = np.nan
    # Numerical rank of the data Jacobian with columns in pixel-equivalent units.
    nd = 2 * len(Pu)
    colscale = np.linalg.norm(J[nd:nd + 7], axis=0) / lam + 1e-12
    sv = np.linalg.svd(J[:nd] / colscale, compute_uv=False)
    rank = int(np.sum(sv > 1e-4 * max(sv.max(), 1e-12)))
    status = ("under-constrained" if rank < dof or nd < dof else
              "solved" if nd == dof else "over-constrained")
    info.update(status=status, rank=rank, errors=err, rms=float(np.sqrt(np.mean(err[use] ** 2))))
    if polish > 0:
        return solve(rig, P, UV, free, q, q, weight, lock_roll, focal_range, iters, polish - 1)
    return q, info


# --------------------------------------------------------------------------------------------
# Pin storage (hidden JSON string parm "pins" on the HDA node)
# --------------------------------------------------------------------------------------------
def load_pins(node):
    try:
        data = json.loads(node.parm("pins").unexpandedString() or "{}")
    except ValueError:
        data = {}
    data.setdefault("version", 1)
    data.setdefault("next_id", 1)
    data.setdefault("frames", {})
    return data


def save_pins(node, data):
    data["frames"] = {k: v for k, v in data["frames"].items() if v}
    node.parm("pins").set(json.dumps(data, separators=(",", ":"), sort_keys=True))


def frame_key(frame=None):
    return str(int(round(hou.frame() if frame is None else frame)))


def pins_at(data, frame=None):
    return data["frames"].get(frame_key(frame), [])


def new_pin(data, frame, p, uv):
    pin = {"id": data["next_id"], "p": [float(v) for v in p], "uv": [float(v) for v in uv],
           "on": True, "lock": False}
    data["next_id"] += 1
    data["frames"].setdefault(frame_key(frame), []).append(pin)
    return pin


def pin_frames(data):
    return sorted(int(k) for k, v in data["frames"].items() if v)


def neighbour_frames(data, frame=None):
    """(previous, next) frame that has pins, or None."""
    f = int(frame_key(frame))
    frames = pin_frames(data)
    return (max([x for x in frames if x < f], default=None), min([x for x in frames if x > f], default=None))


def nearest_pin_frame(data, frame=None):
    prev, nxt = neighbour_frames(data, frame)
    f = int(frame_key(frame))
    cands = [x for x in (prev, nxt) if x is not None]
    return min(cands, key=lambda x: (abs(x - f), x > f)) if cands else None


def edit_pins(node, label, fn):
    """Load pins, run fn(data) (mutates it), save. One undo step. Returns fn's result."""
    with hou.undos.group(UNDO_PREFIX + label):
        data = load_pins(node)
        result = fn(data)
        save_pins(node, data)
    return result


# --------------------------------------------------------------------------------------------
# Camera access
# --------------------------------------------------------------------------------------------
def target_camera(node):
    cam = node.parm("camera").evalAsNode()
    return cam if cam is not None and cam.type().name() == "cam" else None


def reference_object(node):
    obj = node.parm("refgeo").evalAsNode()
    return obj if isinstance(obj, hou.ObjNode) and obj.displayNode() is not None else None


def camera_problems(cam):
    """-> (fatal reason or None, {parm: reason} for solve parms that must not be written)."""
    if cam is None:
        return "no target camera (set 'Target Camera' to a camera object)", {}
    if cam.isInsideLockedHDA():
        return "camera is inside a locked asset", {}
    if cam.parm("lookatpath").evalAsString():
        return "camera uses Look At", {}
    if cam.parm("constraints_on").eval():
        return "camera uses constraints", {}
    if cam.parm("projection").evalAsString() != "perspective":
        return "camera is not perspective", {}
    protected = {}
    for name in PARMS:
        p = cam.parm(name)
        if p.isLocked():
            protected[name] = "locked"
        elif p.overrideTrack() is not None:
            protected[name] = "CHOP override"
        elif any(k.expressionLanguage() != hou.exprLanguage.Hscript or not _CURVE_FUNCS.match(k.expression().strip())
                 for k in p.keyframes()):
            protected[name] = "expression/reference"
    return None, protected


def free_mask(node, protected=()):
    return [not node.parm("lock_" + n).eval() and n not in protected for n in PARMS]


def solver_options(node):
    return {"weight": node.parm("minchange").eval(), "lock_roll": bool(node.parm("lock_roll").eval()),
            "focal_range": tuple(node.parmTuple("focalrange").eval())}


def write_camera(cam, q, free):
    """Live update: set (or mark pending, if animated) the free solve parms."""
    for name, value, on in zip(PARMS, q, free):
        if on:
            cam.parm(name).setPending(float(value))


def key_camera(cam, q, free, frame=None):
    t = hou.frameToTime(hou.frame() if frame is None else frame)
    for name, value, on in zip(PARMS, q, free):
        if on:
            k = hou.Keyframe(float(value), t)
            k.setExpression("bezier()", hou.exprLanguage.Hscript)   # Houdini's UI default...
            k.setSlopeAuto(True)                                      # ...with auto slopes
            cam.parm(name).setKeyframe(k)


def keyed_frames(cam):
    if cam is None:
        return []
    return sorted({int(round(k.frame())) for n in PARMS for k in cam.parm(n).keyframes()})


def delete_keys(cam, frame=None):
    f = hou.frame() if frame is None else frame
    for name in PARMS:
        p = cam.parm(name)
        if not p.isLocked() and any(abs(k.frame() - f) < 1e-6 for k in p.keyframes()):
            p.deleteKeyframeAtFrame(f)


def solve_pins(node, cam, pins, anchor=None, start=None, rig=None, polish=0):
    """Solve `cam` for the active pins. Does not write. Returns (q or None, info, free, rig)."""
    fatal, protected = camera_problems(cam)
    if fatal:
        return None, {"status": fatal, "rms": 0.0, "n": 0}, [False] * 7, None
    rig = rig or Rig(cam)
    free = free_mask(node, protected)
    act = [p for p in pins if p["on"]]
    q, info = solve(rig, [p["p"] for p in act], [p["uv"] for p in act], free,
                    rig.q if anchor is None else anchor, start, polish=polish, **solver_options(node))
    info["pin_ids"] = [p["id"] for p in act]
    return q, info, free, rig


def solve_and_key(node, key=True):
    """Solve the current frame from the current camera and write (and key) the result."""
    cam = target_camera(node)
    pins = pins_at(load_pins(node))
    q, info, free, _ = solve_pins(node, cam, pins, polish=2)
    if q is not None and info["n"]:
        with hou.undos.group(UNDO_PREFIX + ("Solve & Key" if key else "Solve")):
            write_camera(cam, q, free)
            if key:
                key_camera(cam, q, free)
    return info


def plate_path(node, frame=None):
    p = node.parm("plate")
    return p.evalAtFrame(hou.frame() if frame is None else frame) if p.unexpandedString() else ""


def match_resolution(node, cam):
    """Set camera resolution to the plate's if 'Match Resolution to Plate' is on."""
    path = plate_path(node)
    if not (cam and path and node.parm("matchres").eval()):
        return False
    try:
        res = tuple(int(v) for v in hou.imageResolution(path))
    except hou.Error:
        return False
    if res[0] > 0 and tuple(cam.parmTuple("res").eval()) != res:
        with hou.undos.group(UNDO_PREFIX + "Match Resolution"):
            cam.parmTuple("res").set(res)
        return True
    return False


# --------------------------------------------------------------------------------------------
# Parameter callbacks (buttons on the HDA). Houdini already wraps these in one undo group.
# --------------------------------------------------------------------------------------------
def _message(text, severity=hou.severityType.Message, buttons=("OK",)):
    if hou.isUIAvailable():
        return hou.ui.displayMessage(text, buttons=buttons, severity=severity, title="Camera Pin Matcher")
    print("Camera Pin Matcher:", text)
    return 0


def set_all(node, key, value):
    def fn(data):
        for pins in data["frames"].values():
            for p in pins:
                p[key] = value
    edit_pins(node, "%s all pins" % {"lock": "Lock" if value else "Unlock", "on": "Activate" if value else "Deactivate"}[key], fn)


def delete_all_pins(node, confirm=True):
    data = load_pins(node)
    n = sum(len(v) for v in data["frames"].values())
    if not n or (confirm and _message("Delete all %d pins on all frames?" % n, hou.severityType.Warning,
                                      ("Delete", "Cancel")) != 0):
        return 0
    edit_pins(node, "Delete all pins", lambda d: d["frames"].clear())
    return n


def delete_frame_pins(node, frame=None):
    return edit_pins(node, "Delete pins on frame", lambda d: len(d["frames"].pop(frame_key(frame), [])))


def copy_nearest_pins(node, frame=None):
    """Copy pins (ids, anchors, 2D targets) from the nearest frame with pins. -> source frame."""
    def fn(data):
        src = nearest_pin_frame(data, frame)
        if src is None or pins_at(data, frame):
            return None
        data["frames"][frame_key(frame)] = json.loads(json.dumps(pins_at(data, src)))
        return src
    return edit_pins(node, "Copy pins from nearest frame", fn)


def cb_unlock_all(kwargs):
    set_all(kwargs["node"], "lock", False)


def cb_deactivate_all(kwargs):
    set_all(kwargs["node"], "on", False)


def cb_delete_all(kwargs):
    delete_all_pins(kwargs["node"])


def cb_delete_frame(kwargs):
    delete_frame_pins(kwargs["node"])


def cb_copy_nearest(kwargs):
    if copy_nearest_pins(kwargs["node"]) is None:
        _message("Nothing copied: this frame already has pins, or no other frame has pins.")


def cb_solve_key(kwargs):
    info = solve_and_key(kwargs["node"])
    if not info.get("n"):
        _message("Nothing solved: %s." % info["status"], hou.severityType.Warning)


def cb_delete_keys(kwargs):
    cam = target_camera(kwargs["node"])
    if cam:
        delete_keys(cam)


def cb_preset(kwargs):
    node, which = kwargs["node"], kwargs["parm_name"]
    if which == "preset_focal":
        node.parm("lock_focal").set(1)
    elif which == "preset_nodal":
        for n in ("tx", "ty", "tz"):
            node.parm("lock_" + n).set(1)
    elif which == "preset_roll":
        node.parm("lock_roll").set(1)
    else:
        for n in PARMS + ("roll",):
            node.parm("lock_" + n).set(0)


def cb_keyed_frames(kwargs):
    """Overview of keyed / pinned frames; picking one jumps there."""
    node = kwargs["node"]
    data, keys = load_pins(node), keyed_frames(target_camera(node))
    frames = sorted(set(keys) | set(pin_frames(data)))
    if not frames:
        _message("No keyed frames and no pins yet.")
        return
    rows = ["%5d   %s   %d pins" % (f, "key" if f in keys else "   ", len(pins_at(data, f))) for f in frames]
    if not hou.isUIAvailable():
        print("\n".join(rows))
        return
    pick = hou.ui.selectFromList(rows, exclusive=True, message="Keyed frames (camera keys / pins). Pick one to jump to it.",
                                 title="Camera Pin Matcher", column_header="frame   key   pins")
    if pick:
        hou.setFrame(frames[pick[0]])


def cb_enter(kwargs):
    sv = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
    if sv is not None:
        kwargs["node"].setCurrent(True, clear_all_selected=True)
        sv.enterCurrentNodeState()
