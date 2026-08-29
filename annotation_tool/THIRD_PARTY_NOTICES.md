# Third-Party Notices

This distribution contains modified source subsets from the following projects:

- `model4dgen/L4GM_MoGA`: derived from [NVIDIA Toronto AI Lab L4GM](https://github.com/nv-tlabs/L4GM-official), under Apache License 2.0. The license and HAT-4D change notice are retained at `model4dgen/L4GM_MoGA/LICENSE` and `model4dgen/L4GM_MoGA/MODIFICATIONS.md`.
- `model4dgen/generative-models`: derived from [Stability AI Generative Models](https://github.com/Stability-AI/generative-models), under the MIT License. The license and HAT-4D change notice are retained at `model4dgen/generative-models/LICENSE-CODE` and `model4dgen/generative-models/MODIFICATIONS.md`.
- The LPIPS sources nested under `model4dgen/generative-models/sgm/modules/autoencoding/lpips` retain their BSD-style license files in the `loss` and `model` directories.

SAM 3D Objects, Segment Anything, and Qwen3-VL are downloaded separately by `scripts/install_external_dependencies.py` and remain subject to their respective code and model licenses. Model checkpoints are not included in this repository. Users must review and accept the applicable model terms before downloading weights.

The HAT-4D adapters and original modifications are distributed under the repository's Apache License 2.0 unless a file or bundled subtree states otherwise. Source-code licenses do not grant rights to separately downloaded model weights or datasets.
