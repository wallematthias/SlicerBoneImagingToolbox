from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


TUTORIALS = {
    "iZs-SEPCT3o": "docs/index.md",
    "bJsD-42t7hk": "docs/tools/segmentation-and-contours.md",
    "wtnzl54njQM": "docs/tools/microarchitecture.md",
    "754TJkOADHA": "docs/tools/parosol-fea.md",
    "zuoMNC3o2XA": "docs/tools/timelapsed-hrpqct.md",
    "g0WAxdSCmLA": "docs/tools/mechanoregulation.md",
    "VRQsGUlJ0Ek": "docs/tools/plate-rod-morphometry.md",
    "KMDTtJk_x0s": "docs/tools/batch-processor.md",
}


def test_tutorial_video_hub_is_in_docs_navigation():
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tutorials = ROOT / "docs" / "tutorials" / "index.md"

    assert "Tutorial Videos: tutorials/index.md" in mkdocs
    assert "docs/tutorials/index.md" in readme
    assert tutorials.exists()


def test_tutorial_video_hub_lists_all_published_walkthroughs():
    text = (ROOT / "docs" / "tutorials" / "index.md").read_text(encoding="utf-8")

    for video_id in TUTORIALS:
        assert f"https://www.youtube.com/watch?v={video_id}" in text
        assert f"https://www.youtube.com/embed/{video_id}" in text


def test_relevant_tool_pages_link_their_tutorial_video():
    for video_id, page in TUTORIALS.items():
        text = (ROOT / page).read_text(encoding="utf-8")
        assert f"https://www.youtube.com/watch?v={video_id}" in text
