"""Reload the rebuilt asset in a running GUI Houdini and re-enter the tool on the test scene.

    hython build_hda.py && hython dev/rpc.py dev/reload.py [--scene]

--scene also (re)loads scenes/pinmatch_test.hip*. Development aid only.
"""
import os
import sys

import hou

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
EXT = {hou.licenseCategoryType.Commercial: "", hou.licenseCategoryType.Indie: "lc"}.get(hou.licenseCategory(), "nc")
sv = hou.ui.paneTabOfType(hou.paneTabType.SceneViewer)
sv.setCurrentState("objview")
hda = os.path.join(ROOT, "otls", "camera_pin_matcher.hda" + EXT)
if hda in hou.hda.loadedFiles():
    hou.hda.reloadFile(hda)
else:
    hou.hda.installFile(hda)
if "--scene" in sys.argv:
    hou.hipFile.load(os.path.join(ROOT, "scenes", "pinmatch_test.hip" + EXT), suppress_save_prompt=True,
                     ignore_load_warnings=True)
node = next((n for n in hou.node("/obj").children() if n.type().name() == "pinmatch::camera_pin_matcher::1.0"), None)
if node is not None:
    node.setCurrent(True, clear_all_selected=True)
    sv.enterCurrentNodeState()
print("state:", sv.currentState(), "| viewport camera:", sv.curViewport().camera())
