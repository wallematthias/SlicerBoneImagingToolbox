from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "HRpQCTTools" / "DeriveLabelsHRpQCT" / "DeriveLabelsHRpQCT.py"


def test_label_algebra_module_owns_derive_label_tools():
    source = MODULE.read_text()

    assert 'parent.title = "Mask and Label Algebra"' in source
    assert "Create FEA Input Labels" in source
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


def test_label_algebra_exposes_segment_role_overrides_for_segmentation_nodes():
    source = MODULE.read_text()

    assert "def _segment_combo(self):" in source
    assert "def _mask_selector_row(self, form, label, role, tooltip):" in source
    assert "def _refresh_segment_combo(self, selector, segment_combo, role):" in source
    assert "def _selected_segment_id(self, segment_combo):" in source
    assert "self.materialSegSelector, self.materialSegSegmentCombo = self._mask_selector_row(" in source
    assert "self.materialTrabSelector, self.materialTrabSegmentCombo = self._mask_selector_row(" in source
    assert "self.materialCortSelector, self.materialCortSegmentCombo = self._mask_selector_row(" in source
    assert "self.materialFullSelector, self.materialFullSegmentCombo = self._mask_selector_row(" in source
    assert "selected_segment_id=selected_segment_id" in source
    assert "seg_segment_id=" in source
    assert "trab_segment_id=" in source
    assert "cort_segment_id=" in source
    assert "full_segment_id=" in source


def test_label_algebra_uses_shared_geometry_when_exporting_multiple_segments():
    source = MODULE.read_text()

    assert "def _segmentation_reference_node(self, segmentation_node, roles_and_segment_ids):" in source
    assert "reference_node = self._shared_segmentation_reference_node(" in source
    assert "shared_reference_node = None" in source
    assert "slicer.mrmlScene.RemoveNode(shared_reference_node)" in source
    assert "segment_id = self._segment_id_for_role(" in source
    assert "segment_ids.InsertNextValue(segment_id)" in source
