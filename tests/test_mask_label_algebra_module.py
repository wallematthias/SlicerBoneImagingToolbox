from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "HRpQCTTools" / "DeriveLabelsHRpQCT" / "DeriveLabelsHRpQCT.py"


def test_label_algebra_module_owns_derive_label_tools():
    source = MODULE.read_text()

    assert 'parent.title = "Mask and Label Algebra"' in source
    assert "Create HOM Material Labels" in source
    assert "Generate Missing Mask" in source
    assert "Mask Operations" in source
    assert "Union" in source
    assert "Relabel Nonzero Voxels" in source
    assert "Validate Mask Set" in source
    assert "Count Selected Masks" in source
    assert "create_material_label_volume" in source
    assert "create_missing_mask_volume" in source
    assert "create_boolean_mask_volume" in source


def test_label_algebra_accepts_contouring_segmentation_nodes():
    source = MODULE.read_text()

    assert '"vtkMRMLSegmentationNode"' in source
    assert "ExportSegmentsToLabelmapNode" in source
    assert "selector.setNodeTypes(node_types)" in source
    assert "_segment_id_for_role" in source
    assert "_segment_tag_value(segment, \"HRpQCT.Role\")" in source
    assert "scene_segment_matches_role" in source
    assert "reference_node=reference_node" in source
    assert "if node.IsA(\"vtkMRMLSegmentationNode\")" in source
    assert "full_source = full_mask_node or segmentation_source" in source
    assert "trab_source = trab_mask_node or segmentation_source" in source
    assert "cort_source_node = cort_mask_node or segmentation_source" in source
