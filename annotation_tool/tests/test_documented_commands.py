import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = REPO_ROOT / "annotation_tool"


def test_root_readme_links_annotation_installation() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Annotation Tool" in text
    assert "annotation_tool/README_eng.md" in text
    assert "annotation_tool/README_ch.md" in text
    assert "ui_README.md" in text
    assert "ui_README_ch.md" in text


def test_annotation_guides_cover_installation_and_workflow() -> None:
    english = (TOOL_ROOT / "README_eng.md").read_text(encoding="utf-8")
    chinese = (TOOL_ROOT / "README_ch.md").read_text(encoding="utf-8")

    for text in (english, chinese):
        assert "start_qwenvl.sh" in text
        assert "start_roi_backend.sh" in text
        assert "start_sam3d_backend.sh" in text
        assert "start_frontend.sh" in text
        assert "Total cover" in text
        assert "RESUME_WORKSPACE" in text
        assert "external_dependencies.json" in text
        assert "HAT4D_API_TOKEN" in text
        assert "HAT4D_ALLOWED_ROOTS" in text
        assert "fa46c1d2101ad8cc32e6d3cc4513e1c083f4e2729935467e3d42a57a88a6403a" in text


def test_ui_guides_document_critical_workflow_and_local_images() -> None:
    english = (REPO_ROOT / "docs" / "ui_README.md").read_text(encoding="utf-8")
    chinese = (REPO_ROOT / "docs" / "ui_README_ch.md").read_text(encoding="utf-8")

    required_text = {
        "Begin SAM3D Gen",
        "Pixel Pick Mode",
        "Generate 3D (Current Mask)",
        "Pose opt hard lock world X",
        "Optimize Pose (Mask Align)",
        "Add All Objects & Exit",
        "L4GM_4DGEN",
        "Generate next segment",
        "Total cover",
    }
    for text in (english, chinese):
        assert all(item in text for item in required_text)
        assert "tmp/workspace" not in text

    assert re.search(r"[\u3040-\u30ff\u3400-\u9fff]", english) is None

    for text in (english, chinese):
        image_paths = re.findall(r'<img src="([^"]+)"', text)
        assert len(image_paths) == 4
        for image_path in image_paths:
            resolved = REPO_ROOT / image_path
            assert resolved.is_file(), image_path
            assert resolved.suffix == ".png"


def test_qwen_protocol_is_explicit_and_user_executed() -> None:
    text = (TOOL_ROOT / "docs" / "qwen3_vl_8xh200_test.md").read_text(encoding="utf-8")

    required_text = {
        "User-executed external verification",
        "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7",
        "QWEN_MODEL_PATH",
        "--tp 8",
        "/v1/models",
        "/agent/video_understanding",
        "Expected output",
        "nvidia-smi",
        "kill",
    }
    assert all(item in text for item in required_text)


def test_public_plugin_readmes_do_not_contain_internal_multilingual_notes() -> None:
    for path in (
        TOOL_ROOT / "viewer_plugins" / "sam3d_gen" / "README.md",
        TOOL_ROOT / "viewer_plugins" / "vlm_assistor" / "README.md",
    ):
        text = path.read_text(encoding="utf-8")
        assert re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text) is None, path


def test_documented_bash_blocks_have_valid_syntax() -> None:
    for path in (
        REPO_ROOT / "README.md",
        TOOL_ROOT / "README.md",
        TOOL_ROOT / "README_eng.md",
        TOOL_ROOT / "README_ch.md",
        TOOL_ROOT / "docs" / "qwen3_vl_8xh200_test.md",
    ):
        text = path.read_text(encoding="utf-8")
        blocks = re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL)
        if path.name != "README.md":
            assert blocks, path
        for index, block in enumerate(blocks):
            result = subprocess.run(
                ["bash", "-n"],
                input=block,
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, f"{path} bash block {index}: {result.stderr}"
