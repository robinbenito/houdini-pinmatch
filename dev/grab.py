"""Save a screenshot of the Houdini main window (the GL viewport included).

    hython dev/rpc.py dev/grab.py /tmp/houdini.png

Qt grab of Houdini's own window only - no screen-recording permission needed. Development aid.
"""
import sys

import hou

out = sys.argv[1] if len(sys.argv) > 1 else "houdini.png"
image = hou.qt.mainWindow().grab()
image.save(out)
print("saved", out, image.width(), "x", image.height())
