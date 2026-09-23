"""Start an RPC server inside a graphical Houdini so dev/rpc.py can drive it from a terminal.

    houdini -foreground dev/start_rpc.py            (or paste into Houdini's Python Shell)

Port 18811 (hrpyc default). Development aid only - not part of the tool.
"""
import hrpyc

hrpyc.start_server(port=18811, use_thread=True, quiet=True)
