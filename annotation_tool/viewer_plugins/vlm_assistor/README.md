# VLM Assistor Plugin

This Viser plugin exposes the HAT-4D Human-Agent workflow: temporal scene
understanding, point planning and mask evaluation, SAM3D object validation,
pose/depth adjustment, and 4D consistency checks.

The recommended public integration is the standalone OpenAI-compatible backend
in `backend_server.py`. It accepts frame paths or a frame directory and returns
a normalized temporal scene graph. The private shared-filesystem VLM toolkit is
optional and is not required for the standalone backend.

See the top-level [`annotation_tool/README.md`](../../README.md) for service
commands and the 8xH200 Qwen3-VL protocol.
