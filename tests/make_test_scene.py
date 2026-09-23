"""Build the Camera Pin Matcher test scene and render its plate from a known camera.

    hython tests/make_test_scene.py [--no-render | --quick]   (--quick renders frame 1 only)

Creates scenes/pinmatch_test.hip(lc|nc) with
  /obj/room    simple room + furniture (the reference mesh)
  /obj/scan    the room as a dense noisy mesh (~1M triangles), display off
  /obj/splat   the room as Gaussian splats (Houdini GSplat attributes) with floaters, display off
  /obj/gt_cam  ground-truth camera (animated 1-24), used to render the plate
  /obj/cam1    camera to be solved, deliberately wrong start pose/focal
  /obj/camera_pin_matcher1   the HDA node, wired to cam1/room/plate (if the HDA is built)
and renders scenes/plate/plate.$F4.jpg (Karma CPU) unless --no-render.
"""
import glob
import os
import subprocess
import sys

import hou

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = {hou.licenseCategoryType.Commercial: "", hou.licenseCategoryType.Indie: "lc"}.get(
    hou.licenseCategory(), "nc")
SCENE = os.path.join(ROOT, "scenes", "pinmatch_test.hip" + EXT)
HDA = os.path.join(ROOT, "otls", "camera_pin_matcher.hda" + EXT)

# Ground truth used by tests/test_core.py and the test log (frame 1).
GT = {"t": (1.4, 1.65, 3.6), "r": (-9.0, 14.0, 0.0), "focal": 30.0}
GT_END = {"t": (0.2, 1.5, 3.0), "r": (-6.0, 2.0, 1.5)}
START = {"t": (2.2, 1.3, 4.4), "r": (-4.0, 24.0, -3.0), "focal": 42.0}
FRAMES = (1, 24)
RES = (1280, 720)


def box(geo, name, size, center, rot=(0, 0, 0)):
    b = geo.createNode("box", name)
    b.parmTuple("size").set(size)
    b.parmTuple("t").set(center)
    b.parmTuple("r").set(rot)
    return b


def build_room():
    geo = hou.node("/obj").createNode("geo", "room")
    parts = [
        box(geo, "room_box", (8, 3.2, 10), (0, 1.6, 0)),
        box(geo, "table", (1.6, 0.8, 1.0), (0.9, 0.4, -2.2)),
        box(geo, "cabinet", (1.2, 2.0, 0.6), (-2.6, 1.0, -4.6)),
        box(geo, "pillar", (0.5, 3.2, 0.5), (2.8, 1.6, -3.2)),
        box(geo, "cube", (0.4, 0.4, 0.4), (0.6, 1.0, -2.3), (0, 20, 0)),
    ]
    merge = geo.createNode("merge")
    for i, p in enumerate(parts):
        merge.setInput(i, p)
    color = geo.createNode("attribwrangle", "face_colors")
    color.setInput(0, merge)
    color.parm("class").set("primitive")
    color.parm("snippet").set("v@Cd = hsvtorgb(set(frac(@primnum * 0.618), 0.45, 0.9));")
    out = geo.createNode("null", "OUT")
    out.setInput(0, color)
    out.setDisplayFlag(True)
    out.setRenderFlag(True)
    geo.layoutChildren()
    return geo


def build_scan():
    """The room as a dense, slightly noisy triangle mesh (~1M triangles), like a scan. Display off."""
    geo = hou.node("/obj").createNode("geo", "scan")
    merge = geo.createNode("object_merge")
    merge.parm("objpath1").set("/obj/room/OUT")
    sub = geo.createNode("subdivide")
    sub.setInput(0, merge)
    sub.parm("algorithm").set("osdbilinear")          # keeps the shape and the original corners
    sub.parm("iterations").set(7)
    tri = geo.createNode("divide")
    tri.setInput(0, sub)
    noise = geo.createNode("attribwrangle", "scan_noise")
    noise.setInput(0, tri)
    noise.parm("snippet").set("@P += (noise(@P * 7.0) - 0.5) * 0.004;     // +-2 mm")
    out = geo.createNode("null", "OUT")
    out.setInput(0, noise)
    out.setDisplayFlag(True)
    geo.layoutChildren()
    geo.setDisplayFlag(False)


def build_splat():
    """The room as Gaussian splats with Houdini's GSplat attributes (what the Bake GSplats SOP makes
    of a .ply): flat discs on the surfaces plus faint floaters in the air. Display off."""
    geo = hou.node("/obj").createNode("geo", "splat")
    merge = geo.createNode("object_merge")
    merge.parm("objpath1").set("/obj/room/OUT")
    scatter = geo.createNode("scatter::2.0")
    scatter.setInput(0, merge)
    scatter.parm("npts").set(500000)
    scatter.parm("relaxpoints").set(0)
    discs = geo.createNode("attribwrangle", "gsplat_attribs")
    discs.setInput(0, scatter)
    discs.setInput(1, merge)
    discs.parm("snippet").set("""
int prim; vector uvw;
xyzdist(1, @P, prim, uvw);
v@Cd = prim(1, "Cd", prim);
p@orient = dihedral({0, 0, 1}, normalize(prim_normal(1, prim, uvw)));    // disc normal = surface normal
float s = 0.03 * (0.7 + 0.6 * rand(@ptnum));
v@scale = set(s, s * (0.5 + 0.5 * rand(@ptnum + 7)), s * 0.1);
f@GS_Alpha = 0.6 + 0.4 * rand(@ptnum + 3);
""")
    floaters = geo.createNode("attribwrangle", "floaters")
    floaters.setInput(0, discs)
    floaters.parm("class").set("detail")
    floaters.parm("snippet").set("""
for (int i = 0; i < 400; i++) {
    int pt = addpoint(0, set(fit01(rand(i), -3.5, 3.5), fit01(rand(i + 1000), 0.2, 3.0), fit01(rand(i + 2000), -4.5, 4.5)));
    vector cd = set(rand(i + 3), rand(i + 4), rand(i + 5));
    vector axis = normalize(set(rand(i + 7) - 0.5, rand(i + 8) - 0.5, rand(i + 9) - 0.5));
    vector4 q = quaternion(radians(360 * rand(i + 6)), axis);
    vector sc = set(0.03, 0.03, 0.03) * (0.5 + rand(i + 10));
    setpointattrib(0, "Cd", pt, cd);
    setpointattrib(0, "orient", pt, q);
    setpointattrib(0, "scale", pt, sc);
    setpointattrib(0, "GS_Alpha", pt, 0.15 + 0.3 * rand(i + 11));
}
""")
    out = geo.createNode("null", "OUT")
    out.setInput(0, floaters)
    out.setDisplayFlag(True)
    geo.layoutChildren()
    geo.setDisplayFlag(False)


def camera(name, t, r, focal):
    cam = hou.node("/obj").createNode("cam", name)
    cam.parmTuple("t").set(t)
    cam.parmTuple("r").set(r)
    cam.parm("focal").set(focal)
    cam.parm("aperture").set(36.0)
    cam.parmTuple("res").set(RES)
    return cam


def key(parm, frame, value):
    k = hou.Keyframe(value, hou.frameToTime(frame))
    k.setExpression("bezier()", hou.exprLanguage.Hscript)
    k.setSlopeAuto(True)
    parm.setKeyframe(k)


def build_scene():
    hou.hipFile.clear(suppress_save_prompt=True)
    hou.playbar.setFrameRange(*FRAMES)
    hou.playbar.setPlaybackRange(*FRAMES)
    hou.setFrame(FRAMES[0])
    build_room()
    build_scan()
    build_splat()

    gt = camera("gt_cam", GT["t"], GT["r"], GT["focal"])
    for name, a, b in (("t", GT["t"], GT_END["t"]), ("r", GT["r"], GT_END["r"])):
        for p, va, vb in zip(gt.parmTuple(name), a, b):
            key(p, FRAMES[0], va)
            key(p, FRAMES[1], vb)
    camera("cam1", START["t"], START["r"], START["focal"])

    for name, pos, intensity in (("key_light", (1.5, 2.9, -1.0), 6.0), ("fill_light", (-2.5, 2.9, 2.5), 3.0)):
        light = hou.node("/obj").createNode("hlight::2.0", name)   # inside the closed room
        light.parm("light_type").set("point")
        light.parmTuple("t").set(pos)
        light.parm("light_intensity").set(intensity)
        light.parm("light_exposure").set(2.0)

    rop = hou.node("/out").createNode("karma", "plate_render")
    rop.parm("camera").set(gt.path())
    rop.parm("picture").set("$HIP/plate/plate.$F4.jpg")
    rop.parm("trange").set("normal")
    rop.parmTuple("f").deleteAllKeyframes()
    rop.parmTuple("f").set((FRAMES[0], FRAMES[1], 1))
    rop.parm("engine").set("cpu")
    rop.parm("samplesperpixel").set(12)
    hou.node("/obj").layoutChildren()
    return rop


def add_matcher():
    if not os.path.exists(HDA):
        print("HDA not built yet (%s) - skipping matcher node" % HDA)
        return
    hou.hda.installFile("$HIP/../otls/" + os.path.basename(HDA))
    n = hou.node("/obj").createNode("pinmatch::camera_pin_matcher::1.0", "camera_pin_matcher1")
    n.parm("camera").set("/obj/cam1")
    n.parm("refgeo").set("/obj/room")
    n.parm("plate").set("$HIP/plate/plate.$F4.jpg")
    n.setCurrent(True, clear_all_selected=True)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "scenes", "plate"), exist_ok=True)
    rop = build_scene()
    hou.hipFile.save(SCENE)          # sets $HIP before rendering
    if "--no-render" not in sys.argv:
        rop.render(frame_range=FRAMES[:1] * 2 if "--quick" in sys.argv else FRAMES, verbose=False)
        for jpg in glob.glob(os.path.join(ROOT, "scenes", "plate", "*.jpg")):   # keep the repo small
            subprocess.check_call([hou.text.expandString("$HFS/bin/hoiiotool"), jpg,
                                   "--attrib", "Compression", "jpeg:88", "-o", jpg])
        if not os.listdir(os.path.join(ROOT, "scenes", "render")):
            os.rmdir(os.path.join(ROOT, "scenes", "render"))                   # empty Karma output dir
    add_matcher()
    hou.setFrame(FRAMES[0])
    hou.hipFile.save(SCENE)
    print("saved", SCENE)
