from __future__ import annotations

import numpy as np
import qt
import slicer
import vtk

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from SlicerBoneImagingToolboxLib.timelapsed_scene import scene_segment_matches_role


MODULE_VERSION = "0.1.0"


def _same_shape_or_raise(named_arrays):
    present = [(name, np.asarray(array)) for name, array in named_arrays if array is not None]
    if not present:
        raise ValueError("Select at least one mask.")
    shape = present[0][1].shape
    mismatches = [f"{name}={array.shape}" for name, array in present if array.shape != shape]
    if mismatches:
        raise ValueError(f"Selected masks must have the same shape. Expected {shape}; got {', '.join(mismatches)}.")
    return shape


def derive_compartment_mask_arrays(*, full=None, trab=None, cort=None, output_role=None):
    provided = {
        "full": None if full is None else np.asarray(full, dtype=bool),
        "trab": None if trab is None else np.asarray(trab, dtype=bool),
        "cort": None if cort is None else np.asarray(cort, dtype=bool),
    }
    _same_shape_or_raise(provided.items())
    provided_roles = [role for role, array in provided.items() if array is not None]
    if len(provided_roles) < 2:
        raise ValueError("Select any two of full, trabecular, and cortical masks.")

    missing_roles = [role for role, array in provided.items() if array is None]
    if output_role in (None, "auto"):
        output_role = missing_roles[0] if len(missing_roles) == 1 else "none"
    output_role = str(output_role)
    if output_role not in {"full", "trab", "cort", "none"}:
        raise ValueError(f"Unsupported output mask role: {output_role}")

    full_array = provided["full"]
    trab_array = provided["trab"]
    cort_array = provided["cort"]

    if full_array is None:
        full_array = trab_array | cort_array
    if trab_array is None:
        trab_array = full_array & ~cort_array
    if cort_array is None:
        cort_array = full_array & ~trab_array

    return {
        "full": full_array.astype(bool, copy=False),
        "trab": trab_array.astype(bool, copy=False),
        "cort": cort_array.astype(bool, copy=False),
        "derived_role": output_role,
    }


def validate_compartment_mask_arrays(*, full=None, trab=None, cort=None):
    masks = derive_compartment_mask_arrays(full=full, trab=trab, cort=cort, output_role="auto")
    full_array = masks["full"]
    trab_array = masks["trab"]
    cort_array = masks["cort"]
    union = trab_array | cort_array
    overlap = trab_array & cort_array
    outside = union & ~full_array
    missing = full_array & ~union
    return {
        "full": int(np.count_nonzero(full_array)),
        "trab": int(np.count_nonzero(trab_array)),
        "cort": int(np.count_nonzero(cort_array)),
        "overlap": int(np.count_nonzero(overlap)),
        "outside_full": int(np.count_nonzero(outside)),
        "full_not_compartment": int(np.count_nonzero(missing)),
        "valid": bool(not np.any(overlap) and not np.any(outside) and not np.any(missing)),
    }


def binary_mask_operation_arrays(mask_a, mask_b, operation):
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    _same_shape_or_raise([("mask_a", mask_a), ("mask_b", mask_b)])
    operation = str(operation)
    if operation == "union":
        return mask_a | mask_b
    if operation == "intersection":
        return mask_a & mask_b
    if operation == "difference":
        return mask_a & ~mask_b
    if operation == "xor":
        return np.logical_xor(mask_a, mask_b)
    raise ValueError(f"Unsupported mask operation: {operation}")


def relabel_nonzero_array(array, label):
    label = int(label)
    if label < 1:
        raise ValueError("Output label must be greater than zero.")
    dtype = np.uint8 if label <= 255 else np.uint16
    relabelled = np.zeros(np.asarray(array).shape, dtype=dtype)
    relabelled[np.asarray(array) > 0] = label
    return relabelled


def material_labels_from_arrays(seg, trab, cort, *, trab_label=100, cort_label=127, cort_source="cort_mask"):
    seg = np.asarray(seg, dtype=bool)
    trab = np.asarray(trab, dtype=bool)
    cort = np.asarray(cort, dtype=bool)
    _same_shape_or_raise([("bone segmentation", seg), ("trabecular mask", trab), ("cortical mask", cort)])
    material = np.zeros(seg.shape, dtype=np.uint8)
    material[seg & trab] = int(trab_label)
    material[seg & cort] = int(cort_label)
    return material, {
        "trab": int(np.count_nonzero(material == int(trab_label))),
        "cort": int(np.count_nonzero(material == int(cort_label))),
        "cort_source": str(cort_source),
    }


class DeriveLabelsHRpQCT(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Mask and Label Algebra"
        parent.categories = ["Bone Imaging.Microstructural Analysis"]
        parent.index = 30
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            "Derive masks and material labelmaps from existing bone segmentation and ROI masks. "
            f"Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = "Author: Matthias Walle. Part of the Bone Imaging Toolbox for 3D Slicer."


class DeriveLabelsHRpQCTLogic(ScriptedLoadableModuleLogic):
    def _mask_array_from_node(self, node, role=None, *, reference_node=None, selected_segment_id=None):
        if node is None:
            return None
        if node.IsA("vtkMRMLSegmentationNode"):
            labelmap_node = self._segmentation_node_to_labelmap(
                node,
                role,
                reference_node=reference_node,
                selected_segment_id=selected_segment_id,
            )
            try:
                return np.asarray(slicer.util.arrayFromVolume(labelmap_node)) > 0
            finally:
                slicer.mrmlScene.RemoveNode(labelmap_node)
        return np.asarray(slicer.util.arrayFromVolume(node)) > 0

    def _array_from_node(self, node, role=None, *, reference_node=None, selected_segment_id=None):
        if node is None:
            return None
        if node.IsA("vtkMRMLSegmentationNode"):
            labelmap_node = self._segmentation_node_to_labelmap(
                node,
                role,
                reference_node=reference_node,
                selected_segment_id=selected_segment_id,
            )
            try:
                return np.asarray(slicer.util.arrayFromVolume(labelmap_node))
            finally:
                slicer.mrmlScene.RemoveNode(labelmap_node)
        return np.asarray(slicer.util.arrayFromVolume(node))

    def _segment_tag_value(self, segment, tag_name):
        if segment is None or not hasattr(segment, "GetTag"):
            return ""
        try:
            tag_value = vtk.mutable("")
            if segment.GetTag(str(tag_name), tag_value):
                if hasattr(tag_value, "get"):
                    return str(tag_value.get())
                return str(tag_value)
            return ""
        except TypeError:
            return str(segment.GetTag(str(tag_name)) or "")

    def _segment_id_for_role(self, segmentation_node, role, selected_segment_id=None):
        if segmentation_node is None or not segmentation_node.IsA("vtkMRMLSegmentationNode"):
            raise ValueError("Select a segmentation node.")
        if selected_segment_id:
            segment = segmentation_node.GetSegmentation().GetSegment(str(selected_segment_id))
            if segment is None:
                raise ValueError(
                    f"Selected segment ID {selected_segment_id} was not found in {segmentation_node.GetName()}."
                )
            return str(selected_segment_id)
        requested_role = str(role or "").strip()
        segmentation = segmentation_node.GetSegmentation()
        if segmentation.GetNumberOfSegments() == 1:
            return segmentation.GetNthSegmentID(0)
        for index in range(segmentation.GetNumberOfSegments()):
            segment_id = segmentation.GetNthSegmentID(index)
            segment = segmentation.GetSegment(segment_id)
            segment_name = str(segment.GetName() if segment is not None else "")
            segment_role = self._segment_tag_value(segment, "HRpQCT.Role")
            if scene_segment_matches_role(segment_name, segment_role, requested_role):
                return segment_id
        raise ValueError(
            f"Could not find a {requested_role or 'matching'} segment in {segmentation_node.GetName()}."
        )

    def _segmentation_node_to_labelmap(self, segmentation_node, role, *, reference_node=None, selected_segment_id=None):
        segment_id = self._segment_id_for_role(
            segmentation_node,
            role,
            selected_segment_id=selected_segment_id,
        )
        labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode",
            f"{segmentation_node.GetName()}_{str(role or 'segment').replace(' ', '_')}",
        )
        segment_ids = vtk.vtkStringArray()
        segment_ids.InsertNextValue(segment_id)
        try:
            if reference_node is not None and reference_node.IsA("vtkMRMLSegmentationNode"):
                reference_node = None
            extent_mode = getattr(slicer.vtkSlicerSegmentationsModuleLogic, "EXTENT_REFERENCE_GEOMETRY", None)
            export_args = [segmentation_node, segment_ids, labelmap_node]
            if reference_node is not None:
                export_args.append(reference_node)
            if reference_node is not None and extent_mode is not None:
                export_args.append(extent_mode)
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(*export_args)
            return labelmap_node
        except Exception:
            slicer.mrmlScene.RemoveNode(labelmap_node)
            raise

    def _labelmap_from_array(self, array, reference_node, name, *, attributes=None):
        if reference_node is None:
            raise ValueError("Select a reference labelmap.")
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", str(name).strip() or "HRpQCT_mask")
        slicer.util.updateVolumeFromArray(node, np.asarray(array))
        if reference_node.IsA("vtkMRMLSegmentationNode"):
            reference_labelmap = self._segmentation_node_to_labelmap(reference_node, "full")
            try:
                node.CopyOrientation(reference_labelmap)
            finally:
                slicer.mrmlScene.RemoveNode(reference_labelmap)
        else:
            node.CopyOrientation(reference_node)
        node.CreateDefaultDisplayNodes()
        for key, value in (attributes or {}).items():
            node.SetAttribute(str(key), str(value))
        return node

    def _first_selected_node(self, *nodes):
        for node in nodes:
            if node is not None:
                return node
        return None

    def _first_available_reference_node(self, *nodes):
        for node in nodes:
            if node is not None and not node.IsA("vtkMRMLSegmentationNode"):
                return node
        return None

    def create_missing_mask_volume(
        self,
        *,
        full_mask_node=None,
        trab_mask_node=None,
        cort_mask_node=None,
        full_segment_id=None,
        trab_segment_id=None,
        cort_segment_id=None,
        output_role="auto",
        output_name="HRpQCT_derived_mask",
    ):
        reference_node = self._first_available_reference_node(full_mask_node, trab_mask_node, cort_mask_node)
        masks = derive_compartment_mask_arrays(
            full=self._mask_array_from_node(
                full_mask_node,
                "full",
                reference_node=reference_node,
                selected_segment_id=full_segment_id,
            ),
            trab=self._mask_array_from_node(
                trab_mask_node,
                "trab",
                reference_node=reference_node,
                selected_segment_id=trab_segment_id,
            ),
            cort=self._mask_array_from_node(
                cort_mask_node,
                "cort",
                reference_node=reference_node,
                selected_segment_id=cort_segment_id,
            ),
            output_role=output_role,
        )
        role = masks["derived_role"]
        if role == "none":
            raise ValueError("Choose which mask to generate when all three compartment masks are selected.")
        reference = self._first_selected_node(full_mask_node, trab_mask_node, cort_mask_node)
        node = self._labelmap_from_array(
            masks[role].astype(np.uint8),
            reference,
            output_name or f"HRpQCT_{role}_derived",
            attributes={
                "HRpQCT.MaskRole": role,
                "HRpQCT.MaskDerived": "1",
            },
        )
        return node, {"role": role, "voxels": int(np.count_nonzero(masks[role]))}

    def validate_compartment_masks(
        self,
        *,
        full_mask_node=None,
        trab_mask_node=None,
        cort_mask_node=None,
        full_segment_id=None,
        trab_segment_id=None,
        cort_segment_id=None,
    ):
        reference_node = self._first_available_reference_node(full_mask_node, trab_mask_node, cort_mask_node)
        return validate_compartment_mask_arrays(
            full=self._mask_array_from_node(
                full_mask_node,
                "full",
                reference_node=reference_node,
                selected_segment_id=full_segment_id,
            ),
            trab=self._mask_array_from_node(
                trab_mask_node,
                "trab",
                reference_node=reference_node,
                selected_segment_id=trab_segment_id,
            ),
            cort=self._mask_array_from_node(
                cort_mask_node,
                "cort",
                reference_node=reference_node,
                selected_segment_id=cort_segment_id,
            ),
        )

    def create_boolean_mask_volume(
        self,
        mask_a_node,
        mask_b_node,
        operation,
        output_name="HRpQCT_mask_operation",
        *,
        mask_a_segment_id=None,
        mask_b_segment_id=None,
    ):
        if mask_a_node is None or mask_b_node is None:
            raise ValueError("Select both input masks.")
        result = binary_mask_operation_arrays(
            self._mask_array_from_node(
                mask_a_node,
                "full",
                reference_node=mask_b_node,
                selected_segment_id=mask_a_segment_id,
            ),
            self._mask_array_from_node(
                mask_b_node,
                "full",
                reference_node=mask_a_node,
                selected_segment_id=mask_b_segment_id,
            ),
            operation,
        )
        node = self._labelmap_from_array(
            result.astype(np.uint8),
            mask_a_node,
            output_name or f"HRpQCT_{operation}",
            attributes={"HRpQCT.MaskOperation": str(operation)},
        )
        return node, {"voxels": int(np.count_nonzero(result)), "operation": str(operation)}

    def relabel_mask_volume(self, source_node, label, output_name="HRpQCT_relabelled", *, source_segment_id=None):
        if source_node is None:
            raise ValueError("Select a source mask.")
        result = relabel_nonzero_array(
            self._array_from_node(source_node, "full", selected_segment_id=source_segment_id),
            int(label),
        )
        node = self._labelmap_from_array(
            result,
            source_node,
            output_name or "HRpQCT_relabelled",
            attributes={"HRpQCT.RelabelValue": int(label)},
        )
        return node, {"voxels": int(np.count_nonzero(result)), "label": int(label)}

    def mask_voxel_counts(self, segment_ids=None, **nodes):
        counts = {}
        segment_ids = segment_ids or {}
        for role, node in nodes.items():
            if node is not None:
                counts[role] = int(
                    np.count_nonzero(
                        self._array_from_node(node, role, selected_segment_id=segment_ids.get(role))
                    )
                )
        if not counts:
            raise ValueError("Select at least one mask.")
        return counts

    def create_material_label_volume(
        self,
        bone_segmentation_node,
        trab_mask_node=None,
        cort_mask_node=None,
        full_mask_node=None,
        *,
        seg_segment_id=None,
        trab_segment_id=None,
        cort_segment_id=None,
        full_segment_id=None,
        trab_label=100,
        cort_label=127,
        output_name="HRpQCT_HOM_material_labels",
    ):
        if bone_segmentation_node is None:
            raise ValueError("Select a bone segmentation labelmap.")

        reference_node = self._first_available_reference_node(
            bone_segmentation_node,
            trab_mask_node,
            cort_mask_node,
            full_mask_node,
        )
        segmentation_source = bone_segmentation_node if bone_segmentation_node.IsA("vtkMRMLSegmentationNode") else None
        full_source = full_mask_node or segmentation_source
        trab_source = trab_mask_node or segmentation_source
        cort_source_node = cort_mask_node or segmentation_source

        seg = self._mask_array_from_node(
            bone_segmentation_node,
            "seg",
            reference_node=reference_node,
            selected_segment_id=seg_segment_id,
        )
        masks = derive_compartment_mask_arrays(
            full=self._mask_array_from_node(
                full_source,
                "full",
                reference_node=reference_node,
                selected_segment_id=full_segment_id,
            ),
            trab=self._mask_array_from_node(
                trab_source,
                "trab",
                reference_node=reference_node,
                selected_segment_id=trab_segment_id,
            ),
            cort=self._mask_array_from_node(
                cort_source_node,
                "cort",
                reference_node=reference_node,
                selected_segment_id=cort_segment_id,
            ),
            output_role="auto",
        )
        if seg.shape != masks["trab"].shape:
            raise ValueError(
                f"Bone segmentation shape {seg.shape} does not match compartment mask shape {masks['trab'].shape}."
            )
        cort_source = (
            "cort_mask"
            if cort_source_node is not None
            else "derived_from_full_minus_trab"
            if full_source is not None and trab_source is not None
            else "derived_from_full_minus_trab"
        )

        material, counts = material_labels_from_arrays(
            seg,
            masks["trab"],
            masks["cort"],
            trab_label=int(trab_label),
            cort_label=int(cort_label),
            cort_source=cort_source,
        )
        if not np.any(material):
            raise ValueError("The selected segmentation and compartment masks do not overlap.")

        node = self._labelmap_from_array(
            material,
            self._first_selected_node(bone_segmentation_node, trab_mask_node, cort_mask_node, full_mask_node),
            output_name or "HRpQCT_HOM_material_labels",
            attributes={
                "HRpQCT.MaterialLabels": "HOM",
                "HRpQCT.TrabLabel": int(trab_label),
                "HRpQCT.CortLabel": int(cort_label),
            },
        )
        return node, counts


class DeriveLabelsHRpQCTWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = DeriveLabelsHRpQCTLogic()
        self._build_ui()
        self.layout.addStretch(1)

    def _tip(self, widget, text):
        widget.toolTip = str(text)
        return widget

    def _labelmap_selector(self):
        selector = slicer.qMRMLNodeComboBox()
        node_types = ["vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode", "vtkMRMLSegmentationNode"]
        selector.nodeTypes = node_types
        if hasattr(selector, "setNodeTypes"):
            selector.setNodeTypes(node_types)
        selector.selectNodeUponCreation = False
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.noneEnabled = True
        selector.setMRMLScene(slicer.mrmlScene)
        return selector

    def _segment_combo(self):
        combo = qt.QComboBox()
        combo.addItem("Auto", "")
        combo.enabled = False
        self._tip(combo, "Segment to use when the selected node is a Slicer segmentation.")
        return combo

    def _mask_selector_row(self, form, label, role, tooltip):
        selector = self._labelmap_selector()
        segment_combo = self._segment_combo()
        self._tip(selector, tooltip)

        row_widget = qt.QWidget()
        row_layout = qt.QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(selector, 2)
        row_layout.addWidget(qt.QLabel("Segment"))
        row_layout.addWidget(segment_combo, 1)
        form.addRow(label, row_widget)

        selector.currentNodeChanged.connect(
            lambda _node, active_selector=selector, active_combo=segment_combo, active_role=role: self._refresh_segment_combo(
                active_selector,
                active_combo,
                active_role,
            )
        )
        return selector, segment_combo

    def _refresh_segment_combo(self, selector, segment_combo, role):
        segment_combo.blockSignals(True)
        segment_combo.clear()
        segment_combo.addItem("Auto", "")
        node = selector.currentNode()
        is_segmentation = bool(node is not None and node.IsA("vtkMRMLSegmentationNode"))
        segment_combo.enabled = is_segmentation
        if is_segmentation:
            try:
                auto_id = self.logic._segment_id_for_role(node, role)
            except Exception:
                auto_id = None
            segmentation = node.GetSegmentation()
            for index in range(segmentation.GetNumberOfSegments()):
                segment_id = segmentation.GetNthSegmentID(index)
                segment = segmentation.GetSegment(segment_id)
                segment_name = str(segment.GetName() if segment is not None else segment_id)
                label = segment_name
                if auto_id and segment_id == auto_id:
                    label = f"{segment_name} (auto)"
                segment_combo.addItem(label, segment_id)
        segment_combo.blockSignals(False)

    def _selected_segment_id(self, segment_combo):
        selected_segment_id = str(segment_combo.currentData or "").strip()
        return selected_segment_id or None

    def _build_ui(self):
        self.messageLabel = qt.QLabel()
        self.messageLabel.wordWrap = True
        self.layout.addWidget(self.messageLabel)

        form = qt.QFormLayout()
        self.layout.addLayout(form)
        self.materialSegSelector, self.materialSegSegmentCombo = self._mask_selector_row(
            form,
            "Bone segmentation",
            "seg",
            "Bone segmentation labelmap used to restrict material labels to segmented bone voxels.",
        )
        self.materialTrabSelector, self.materialTrabSegmentCombo = self._mask_selector_row(
            form,
            "Trabecular mask",
            "trab",
            "Trabecular ROI mask.",
        )
        self.materialCortSelector, self.materialCortSegmentCombo = self._mask_selector_row(
            form,
            "Cortical mask",
            "cort",
            "Cortical ROI mask.",
        )
        self.materialFullSelector, self.materialFullSegmentCombo = self._mask_selector_row(
            form,
            "Full mask",
            "full",
            "Full/periosteal ROI mask.",
        )

        missing_box = qt.QGroupBox("Derive Missing Compartment Mask")
        missing_form = qt.QFormLayout(missing_box)
        self.missingMaskRoleCombo = qt.QComboBox()
        for label, value in [("Auto", "auto"), ("Full", "full"), ("Trabecular", "trab"), ("Cortical", "cort")]:
            self.missingMaskRoleCombo.addItem(label, value)
        self.missingMaskOutputNameEdit = qt.QLineEdit("HRpQCT_derived_mask")
        self.generateMissingMaskButton = qt.QPushButton("Generate Missing Mask")
        self.generateMissingMaskButton.clicked.connect(self._generate_missing_mask)
        missing_form.addRow("Output role", self.missingMaskRoleCombo)
        missing_form.addRow("Output name", self.missingMaskOutputNameEdit)
        missing_form.addRow(self.generateMissingMaskButton)
        self.layout.addWidget(missing_box)

        hom_box = qt.QGroupBox("HOM Material Labels")
        hom_form = qt.QFormLayout(hom_box)
        self.materialTrabLabelSpin = qt.QSpinBox()
        self.materialTrabLabelSpin.minimum = 1
        self.materialTrabLabelSpin.maximum = 255
        self.materialTrabLabelSpin.value = 100
        self.materialCortLabelSpin = qt.QSpinBox()
        self.materialCortLabelSpin.minimum = 1
        self.materialCortLabelSpin.maximum = 255
        self.materialCortLabelSpin.value = 127
        self.materialOutputNameEdit = qt.QLineEdit("HRpQCT_HOM_material_labels")
        self.createMaterialLabelsButton = qt.QPushButton("Create HOM Material Labels")
        self.createMaterialLabelsButton.clicked.connect(self._create_material_labels)
        hom_form.addRow("Trab label", self.materialTrabLabelSpin)
        hom_form.addRow("Cort label", self.materialCortLabelSpin)
        hom_form.addRow("Output name", self.materialOutputNameEdit)
        hom_form.addRow(self.createMaterialLabelsButton)
        self.layout.addWidget(hom_box)

        operations_box = qt.QGroupBox("Mask Operations")
        operations_form = qt.QFormLayout(operations_box)
        self.maskASelector, self.maskASegmentCombo = self._mask_selector_row(
            operations_form,
            "Mask A",
            "full",
            "First mask for the boolean operation.",
        )
        self.maskBSelector, self.maskBSegmentCombo = self._mask_selector_row(
            operations_form,
            "Mask B",
            "full",
            "Second mask for the boolean operation.",
        )
        self.maskOperationCombo = qt.QComboBox()
        for label, value in [("Union", "union"), ("Intersection", "intersection"), ("A minus B", "difference"), ("XOR", "xor")]:
            self.maskOperationCombo.addItem(label, value)
        self.maskOperationOutputNameEdit = qt.QLineEdit("HRpQCT_mask_operation")
        self.createMaskOperationButton = qt.QPushButton("Create Mask Operation")
        self.createMaskOperationButton.clicked.connect(self._create_mask_operation)
        operations_form.addRow("Operation", self.maskOperationCombo)
        operations_form.addRow("Output name", self.maskOperationOutputNameEdit)
        operations_form.addRow(self.createMaskOperationButton)
        self.layout.addWidget(operations_box)

        relabel_box = qt.QGroupBox("Relabel And Validate")
        relabel_form = qt.QFormLayout(relabel_box)
        self.relabelSourceSelector, self.relabelSourceSegmentCombo = self._mask_selector_row(
            relabel_form,
            "Source",
            "full",
            "Source mask to relabel or count.",
        )
        self.relabelValueSpin = qt.QSpinBox()
        self.relabelValueSpin.minimum = 1
        self.relabelValueSpin.maximum = 65535
        self.relabelValueSpin.value = 126
        self.relabelOutputNameEdit = qt.QLineEdit("HRpQCT_relabelled")
        self.relabelButton = qt.QPushButton("Relabel Nonzero Voxels")
        self.relabelButton.clicked.connect(self._relabel_mask)
        self.validateMasksButton = qt.QPushButton("Validate Mask Set")
        self.validateMasksButton.clicked.connect(self._validate_mask_set)
        self.countMasksButton = qt.QPushButton("Count Selected Masks")
        self.countMasksButton.clicked.connect(self._count_selected_masks)
        relabel_form.addRow("Label", self.relabelValueSpin)
        relabel_form.addRow("Output name", self.relabelOutputNameEdit)
        relabel_form.addRow(self.relabelButton)
        relabel_form.addRow(self.validateMasksButton)
        relabel_form.addRow(self.countMasksButton)
        self.layout.addWidget(relabel_box)

    def _log(self, text):
        self.messageLabel.text = str(text)

    def _error(self, exc):
        slicer.util.errorDisplay(str(exc))
        self._log(f"Error: {exc}")

    def _generate_missing_mask(self):
        try:
            node, counts = self.logic.create_missing_mask_volume(
                full_mask_node=self.materialFullSelector.currentNode(),
                trab_mask_node=self.materialTrabSelector.currentNode(),
                cort_mask_node=self.materialCortSelector.currentNode(),
                full_segment_id=self._selected_segment_id(self.materialFullSegmentCombo),
                trab_segment_id=self._selected_segment_id(self.materialTrabSegmentCombo),
                cort_segment_id=self._selected_segment_id(self.materialCortSegmentCombo),
                output_role=str(self.missingMaskRoleCombo.currentData),
                output_name=self.missingMaskOutputNameEdit.text.strip() or "HRpQCT_derived_mask",
            )
            self._log(f"Created {node.GetName()}. Role={counts['role']}, voxels={counts['voxels']}.")
        except Exception as exc:
            self._error(exc)

    def _create_material_labels(self):
        try:
            node, counts = self.logic.create_material_label_volume(
                self.materialSegSelector.currentNode(),
                self.materialTrabSelector.currentNode(),
                self.materialCortSelector.currentNode(),
                self.materialFullSelector.currentNode(),
                seg_segment_id=self._selected_segment_id(self.materialSegSegmentCombo),
                trab_segment_id=self._selected_segment_id(self.materialTrabSegmentCombo),
                cort_segment_id=self._selected_segment_id(self.materialCortSegmentCombo),
                full_segment_id=self._selected_segment_id(self.materialFullSegmentCombo),
                trab_label=int(self.materialTrabLabelSpin.value),
                cort_label=int(self.materialCortLabelSpin.value),
                output_name=self.materialOutputNameEdit.text.strip() or "HRpQCT_HOM_material_labels",
            )
            self._log(
                f"Created {node.GetName()}. Material voxels: "
                f"trab={counts.get('trab')}, cort={counts.get('cort')} ({counts.get('cort_source')})."
            )
        except Exception as exc:
            self._error(exc)

    def _create_mask_operation(self):
        try:
            node, counts = self.logic.create_boolean_mask_volume(
                self.maskASelector.currentNode(),
                self.maskBSelector.currentNode(),
                str(self.maskOperationCombo.currentData),
                output_name=self.maskOperationOutputNameEdit.text.strip() or "HRpQCT_mask_operation",
                mask_a_segment_id=self._selected_segment_id(self.maskASegmentCombo),
                mask_b_segment_id=self._selected_segment_id(self.maskBSegmentCombo),
            )
            self._log(f"Created {node.GetName()}. Operation={counts['operation']}, voxels={counts['voxels']}.")
        except Exception as exc:
            self._error(exc)

    def _relabel_mask(self):
        try:
            node, counts = self.logic.relabel_mask_volume(
                self.relabelSourceSelector.currentNode(),
                int(self.relabelValueSpin.value),
                output_name=self.relabelOutputNameEdit.text.strip() or "HRpQCT_relabelled",
                source_segment_id=self._selected_segment_id(self.relabelSourceSegmentCombo),
            )
            self._log(f"Created {node.GetName()}. Label={counts['label']}, voxels={counts['voxels']}.")
        except Exception as exc:
            self._error(exc)

    def _validate_mask_set(self):
        try:
            counts = self.logic.validate_compartment_masks(
                full_mask_node=self.materialFullSelector.currentNode(),
                trab_mask_node=self.materialTrabSelector.currentNode(),
                cort_mask_node=self.materialCortSelector.currentNode(),
                full_segment_id=self._selected_segment_id(self.materialFullSegmentCombo),
                trab_segment_id=self._selected_segment_id(self.materialTrabSegmentCombo),
                cort_segment_id=self._selected_segment_id(self.materialCortSegmentCombo),
            )
            status = "valid" if counts["valid"] else "not valid"
            self._log(
                f"Mask set {status}. full={counts['full']}, trab={counts['trab']}, cort={counts['cort']}, "
                f"overlap={counts['overlap']}, outside_full={counts['outside_full']}, "
                f"full_not_compartment={counts['full_not_compartment']}."
            )
        except Exception as exc:
            self._error(exc)

    def _count_selected_masks(self):
        try:
            counts = self.logic.mask_voxel_counts(
                segment_ids={
                    "seg": self._selected_segment_id(self.materialSegSegmentCombo),
                    "full": self._selected_segment_id(self.materialFullSegmentCombo),
                    "trab": self._selected_segment_id(self.materialTrabSegmentCombo),
                    "cort": self._selected_segment_id(self.materialCortSegmentCombo),
                },
                seg=self.materialSegSelector.currentNode(),
                full=self.materialFullSelector.currentNode(),
                trab=self.materialTrabSelector.currentNode(),
                cort=self.materialCortSelector.currentNode(),
            )
            self._log("Voxel counts: " + ", ".join(f"{role}={count}" for role, count in counts.items()) + ".")
        except Exception as exc:
            self._error(exc)


class DeriveLabelsHRpQCTTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.test_array_helpers()

    def test_array_helpers(self):
        full = np.array([[[1, 1], [1, 0]]], dtype=bool)
        cort = np.array([[[1, 0], [0, 0]]], dtype=bool)
        masks = derive_compartment_mask_arrays(full=full, cort=cort, output_role="trab")
        self.assertEqual(int(np.count_nonzero(masks["trab"])), 2)
        validation = validate_compartment_mask_arrays(full=full, trab=masks["trab"], cort=cort)
        self.assertTrue(validation["valid"])
