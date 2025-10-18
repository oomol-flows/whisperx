#!/bin/bash
export LD_LIBRARY_PATH="/root/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/site-packages/nvidia/cudnn/lib:/root/.local/share/uv/python/cpython-3.11.13-linux-x86_64-gnu/lib/python3.11/site-packages/ctranslate2.libs:$LD_LIBRARY_PATH"
exec python __init__.py "$@"
