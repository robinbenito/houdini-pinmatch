"""Build otls/camera_pin_matcher.hda from src/.   Usage:  hython build_hda.py

The asset is a build product: src/pinmatch.py becomes the HDA PythonModule, src/pinmatch_state.py
the embedded ViewerStateModule. The file extension follows the license that builds it
(.hda commercial, .hdalc Indie, .hdanc Apprentice/Education).
"""
import os
import shutil

import hou

ROOT = os.path.dirname(os.path.abspath(__file__))
TYPE_NAME = "pinmatch::camera_pin_matcher::1.0"
EXT = {hou.licenseCategoryType.Commercial: "", hou.licenseCategoryType.Indie: "lc"}.get(hou.licenseCategory(), "nc")
HDA_PATH = os.path.join(ROOT, "otls", "camera_pin_matcher.hda" + EXT)
PY = hou.scriptLanguage.Python


def read(name):
    with open(os.path.join(ROOT, "src", name)) as f:
        return f.read()


def button(name, label, callback, **kw):
    return hou.ButtonParmTemplate(name, label, script_callback="hou.phm().%s(kwargs)" % callback,
                                  script_callback_language=PY, **kw)


def toggle(name, label, default=False, **kw):
    return hou.ToggleParmTemplate(name, label, default_value=default, **kw)


def menu(name, label, items, labels, default=0):
    return hou.MenuParmTemplate(name, label, items, labels, default_value=default)


def parm_templates():
    lock = [toggle("lock_" + n, lbl, join_with_next=True)
            for n, lbl in (("tx", "TX"), ("ty", "TY"), ("tz", "TZ"), ("rx", "RX"), ("ry", "RY"), ("rz", "RZ"),
                           ("focal", "Focal"))]
    lock.append(toggle("lock_roll", "Roll (horizon)"))
    lock[0].setLabel("Lock   TX")
    setup = [
        hou.StringParmTemplate("camera", "Target Camera", 1, string_type=hou.stringParmType.NodeReference,
                               tags={"opfilter": "!!OBJ/CAMERA!!", "oprelative": "."},
                               help="The camera the tool solves. Only its tx ty tz rx ry rz focal are written."),
        hou.StringParmTemplate("refgeo", "Reference Geometry", 1, string_type=hou.stringParmType.NodeReference,
                               tags={"opfilter": "!!OBJ/GEOMETRY!!", "oprelative": "."},
                               help="Object whose display geometry pins are anchored to. Never modified."),
        hou.StringParmTemplate("plate", "Plate Image", 1, string_type=hou.stringParmType.FileReference,
                               file_type=hou.fileType.Image, tags={"filechooser_mode": "read"},
                               help="Plate image or sequence, e.g. $HIP/plate/plate.$F4.jpg"),
        toggle("matchres", "Match Camera Resolution to Plate",
               help="Set the camera's resolution to the plate's (on enter and when the plate changes)."),
        button("enter", "Enter Pin Matcher Tool", "cb_enter"),
    ] + [hou.FloatParmTemplate(n, n, 1, default_value=(v,), is_hidden=True)      # tool view: 2D pan/zoom, near clip
         for n, v in (("view_cx", 0.5), ("view_cy", 0.5), ("view_zoom", 1.0), ("view_near", 0.0))] + [
        hou.StringParmTemplate("view_mask", "view_mask", 1, default_value=("",), is_hidden=True)]   # user's mask, see onEnter
    display = [
        menu("platemode", "Plate Mode", ("bg", "fg"), ("Plate Behind Geometry", "Plate Over Geometry")),
        hou.FloatParmTemplate("opacity", "Plate Opacity", 1, default_value=(0.5,), min=0.0, max=1.0,
                              min_is_strict=True, max_is_strict=True, help="Opacity of the plate over the geometry."),
        menu("meshdisplay", "Mesh Display", ("wire", "hidden", "ghost", "shaded"),
             ("Wireframe (All Edges)", "Hidden Line", "Hidden Line Ghost", "Shaded"), default=1),
        hou.FloatParmTemplate("wireopacity", "Wire Opacity", 1, default_value=(0.7,), min=0.0, max=1.0,
                              min_is_strict=True, max_is_strict=True),
        hou.FloatParmTemplate("wirecolor", "Mesh Wire Color", 3, default_value=(0.3, 1.0, 0.5),
                              look=hou.parmLook.ColorSquare, naming_scheme=hou.parmNamingScheme.RGBA),
        toggle("showghosts", "Show Ghost Pins (previous / next pinned frame)", True),
        toggle("showerrors", "Show Per-Pin Error (px)", True),
    ]
    pins = [
        menu("snap", "Snap New Pins To", ("points", "edges", "surface"), ("Points", "Edges", "Free Surface Hit")),
        menu("createmod", "Create Pin Modifier", ("ctrl", "shift", "ctrlshift"), ("Ctrl", "Shift", "Ctrl+Shift")),
        hou.SeparatorParmTemplate("sep1"),
        button("copynearest", "Copy Pins from Nearest Keyed Frame", "cb_copy_nearest"),
        button("unlockall", "Unlock All Pins", "cb_unlock_all"),
        button("deactivateall", "Deactivate All Pins", "cb_deactivate_all"),
        button("deleteframe", "Delete Pins on Current Frame", "cb_delete_frame"),
        button("deleteall", "Delete All Pins...", "cb_delete_all"),
        hou.StringParmTemplate("pins", "Pin Data (JSON)", 1, default_value=("",), is_hidden=True),
    ]
    solver = lock + [
        button("preset_focal", "Lock Focal/FOV", "cb_preset", join_with_next=True),
        button("preset_nodal", "Nodal (Lock Position)", "cb_preset", join_with_next=True),
        button("preset_roll", "Lock Roll", "cb_preset", join_with_next=True),
        button("preset_clear", "Unlock All", "cb_preset"),
        hou.FloatParmTemplate("focalrange", "Focal Range", 2, default_value=(5.0, 1000.0), min=0.1, max=5000.0,
                              help="Solved focal length is clamped to this range (camera focal units)."),
        hou.FloatParmTemplate("minchange", "Minimal Change Weight", 1, default_value=(1.0,), min=0.01, max=100.0,
                              min_is_strict=True, help="Strength of the pull toward the camera's current state. "
                              "Higher = steadier camera with few pins, lower = follow pins more freely."),
        toggle("autokey", "Solve & Key on Mouse Release", True,
               help="Off: dragging updates the camera live but keys are only set by Solve & Key."),
        hou.SeparatorParmTemplate("sep2"),
        button("solvekey", "Solve & Key", "cb_solve_key", join_with_next=True),
        button("deletekeys", "Delete Keys on Current Frame", "cb_delete_keys", join_with_next=True),
        button("keyedframes", "Keyed Frames...", "cb_keyed_frames"),
    ]
    tabs = [("setup", "Setup", setup), ("display", "Display", display), ("pinstab", "Pins", pins),
            ("solver", "Solver", solver)]
    return [hou.FolderParmTemplate("f_" + n, lbl, parm_templates=p, folder_type=hou.folderType.Tabs) for n, lbl, p in tabs]


HELP = """= Camera Pin Matcher =

#type: node
#context: obj
#icon: OBJ/camera

\"\"\"Match a camera to existing geometry with 2D/3D pins over a plate. Only the camera changes.\"\"\"

Select the node, move the mouse over the viewport and press ((Enter)) (or use __Enter Pin Matcher Tool__).
The viewport locks to the target camera. ((Ctrl + LMB)) on the reference mesh creates a pin, drag a pin
onto the matching plate feature, the camera is solved live. See README.md in the tool repository.
"""


def add_view_proxy(sub):
    """Camera inside the asset that mirrors the target camera (world transform + lens) and applies
    the tool's 2D pan/zoom through its own screen window. The viewer state looks through it, so
    viewing never touches the target camera and the view never detaches from it."""
    cam = sub.createNode("cam", "view_proxy")
    C = 'chsop("../camera")'
    tgt = lambda parm, default: 'if(strlen(%s), ch(%s + "/%s"), %s)' % (C, C, parm, default)
    for a in "XYZ":
        cam.parm("t" + a.lower()).setExpression('origin("", %s, "T%s")' % (C, a))
        cam.parm("r" + a.lower()).setExpression('origin("", %s, "R%s")' % (C, a))
    for parm, default in (("focal", 50), ("aperture", 41.4214), ("far", 10000), ("resx", 1920), ("resy", 1080),
                          ("aspect", 1)):
        cam.parm(parm).setExpression(tgt(parm, default))
    cam.parm("near").setExpression('max(%s, ch("../view_near"))' % tgt("near", 0.001))   # depth precision, see _fit_near
    for a in "xy":
        cam.parm("win" + a).setExpression('%s + %s * (ch("../view_c%s") - 0.5)' % (tgt("win" + a, 0), tgt("winsize" + a, 1), a))
        cam.parm("winsize" + a).setExpression('%s / ch("../view_zoom")' % tgt("winsize" + a, 1))
    cam.setDisplayFlag(False)


def build():
    hou.hipFile.clear(suppress_save_prompt=True)
    os.makedirs(os.path.dirname(HDA_PATH), exist_ok=True)
    if os.path.exists(HDA_PATH):
        os.chmod(HDA_PATH, 0o644)        # it was left read-only (below); Windows can't delete it otherwise
        os.remove(HDA_PATH)
    sub = hou.node("/obj").createNode("subnet", "camera_pin_matcher")
    add_view_proxy(sub)
    node = sub.createDigitalAsset(name=TYPE_NAME, hda_file_name=HDA_PATH, description="Camera Pin Matcher",
                                  min_num_inputs=0, max_num_inputs=0, ignore_external_references=True)
    d = node.type().definition()
    d.setParmTemplateGroup(hou.ParmTemplateGroup(parm_templates()))
    # OBJ assets get wrapped in the standard object switcher (Transform, Subnet, <asset>):
    # hide the stock tabs and promote our tabs next to them.
    ptg = d.parmTemplateGroup()
    std = ptg.entries()
    ours = list(std[-1].parmTemplates())
    ptg.remove(std[-1].name())
    for e in ptg.entries():
        ptg.hideFolder(e.label(), True)
    for f in ours:
        ptg.append(f)
    d.setParmTemplateGroup(ptg)
    d.setIcon("OBJ_camera")
    d.setVersion("1.0")
    d.setComment("Camera Pin Matcher - interactive 2D/3D pin camera matching")
    vs = "__import__('viewerstate.utils', fromlist=[None]).%s_pystate_embedded(kwargs['type'])"
    for section, source, extra in (("PythonModule", read("pinmatch.py"), {}),
                                   ("ViewerStateModule", read("pinmatch_state.py"), {"IsViewerState": True}),
                                   ("ViewerStateInstall", vs % "register", {"IsViewerState": True}),
                                   ("ViewerStateUninstall", vs % "unregister", {"IsViewerState": True})):
        d.addSection(section, source)
        opts = {"IsPython": True, "IsScript": True, "IsExpr": False, "Source": ""}
        opts.update(extra)
        for k, v in opts.items():
            d.setExtraFileOption("%s/%s" % (section, k), v)
    d.addSection("DefaultState", TYPE_NAME)
    d.addSection("Help", HELP)
    d.save(HDA_PATH, create_backup=False)
    shutil.rmtree(os.path.join(ROOT, "otls", "backup"), ignore_errors=True)   # Houdini's per-edit backups
    os.chmod(HDA_PATH, 0o444)   # a build product: no Houdini session can save into it or delete the asset from it
    print("built", HDA_PATH)


if __name__ == "__main__":
    build()
