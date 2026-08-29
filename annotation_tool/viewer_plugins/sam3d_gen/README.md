# SAM3D Plugin

This Viser plugin provides point-prompted SAM segmentation, SAM3D Objects
generation, pose refinement, and manual object transforms. GPU inference runs
in the independent HTTP backend under `backend/server.py`.

The external SAM3D source and its gated checkpoints are not distributed in
this repository. Install the locked source checkout with
`scripts/install_external_dependencies.py`, then follow the upstream
`doc/setup.md` and model license. Configure the SAM checkpoint and SAM3D
`pipeline.yaml` explicitly when launching the backend.

See the top-level [`annotation_tool/README.md`](../../README.md) for the exact
installation, health check, backend, frontend, and Human-Agent commands.
