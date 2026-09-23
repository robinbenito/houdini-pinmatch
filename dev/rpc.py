"""Run a Python file (or -c code) inside a running GUI Houdini started with dev/start_rpc.py.

    hython dev/rpc.py dev/reload.py              # rebuild-and-reload loop, see DEVELOPMENT.md
    hython dev/rpc.py -c "print(hou.frame())"

The code runs on Houdini's main thread (UI calls are safe) with sys.argv = [file, args...];
whatever it prints is returned here.
"""
import os
import sys

import rpyc

conn = rpyc.classic.connect("localhost", 18811)
conn._config["sync_request_timeout"] = 600
if sys.argv[1] == "-c":
    code, argv = sys.argv[2], ["<rpc>"] + sys.argv[3:]
else:
    code, argv = open(sys.argv[1]).read(), [os.path.abspath(sys.argv[1])] + sys.argv[2:]

conn.execute(r'''
import hdefereval, hou, io, sys, traceback
def _dev_run(code, argv):
    buf = io.StringIO()
    def job():
        old = sys.stdout, sys.stderr, sys.argv
        sys.stdout = sys.stderr = buf
        sys.argv = list(argv)
        try:
            exec(compile(code, argv[0], "exec"), {"__name__": "__rpc__", "hou": hou})
        except Exception:
            traceback.print_exc()
        finally:
            sys.stdout, sys.stderr, sys.argv = old
    hdefereval.executeInMainThreadWithResult(job)
    return buf.getvalue()
''')
print(conn.namespace["_dev_run"](code, list(argv)), end="")
