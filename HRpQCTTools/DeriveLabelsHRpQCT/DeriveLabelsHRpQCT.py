from __future__ import annotations

import numpy as np
import qt
import slicer

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


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
    def _mask_array_from_node(self, node):
        if node is None:
            return None
        return np.asarray(slicer.util.arrayFromVolume(node)) > 0

    def _labelmap_from_array(self, array, reference_node, name, *, attributes=None):
        if reference_node is None:
            raise ValueError("Select a reference labelmap.")
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", str(name).strip() or "HRpQCT_mask")
        slicer.util.updateVolumeFromArray(node, np.asarray(array))
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

    def create_missing_mask_volume(
        self,
        *,
        full_mask_node=None,
        trab_mask_node=None,
        cort_mask_node=None,
        output_role="auto",
        output_name="HRpQCT_derived_mask",
    ):
        masks = derive_compartment_mask_arrays(
            full=self._mask_array_from_node(full_mask_node),
            trab=self._mask_array_from_node(trab_mask_node),
            cort=self._mask_array_from_node(cort_mask_node),
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

    def validate_compartment_masks(self, *, full_mask_node=None, trab_mask_node=None, cort_mask_node=None):
        return validate_compartment_mask_arrays(
            full=self._mask_array_from_node(full_mask_node),
            trab=self._mask_array_from_node(trab_mask_node),
            cort=self._mask_array_from_node(cort_mask_node),
        )

    def create_boolean_mask_volume(self, mask_a_node, mask_b_node, operation, output_name="HRpQCT_mask_operation"):
        if mask_a_node is None or mask_b_node is None:
            raise ValueError("Select both input masks.")
        result = binary_mask_operation_arrays(
            self._mask_array_from_node(mask_a_node),
            self._mask_array_from_node(mask_b_node),
            operation,
        )
        node = self._labelmap_from_array(
            result.astype(np.uint8),
            mask_a_node,
            output_name or f"HRpQCT_{operation}",
            attributes={"HRpQCT.MaskOperation": str(operation)},
        )
        return node, {"voxels": int(np.count_nonzero(result)), "operation": str(operation)}

    def relabel_mask_volume(self, source_node, label, output_name="HRpQCT_relabelled"):
        if source_node is None:
            raise ValueError("Select a source mask.")
        result = relabel_nonzero_array(slicer.util.arrayFromVolume(source_node), int(label))
        node = self._labelmap_from_array(
            result,
            source_node,
            output_name or "HRpQCT_relabelled",
            attributes={"HRpQCT.RelabelValue": int(label)},
        )
        return node, {"voxels": int(np.count_nonzero(result)), "label": int(label)}

    def mask_voxel_counts(self, **nodes):
        counts = {}
        for role, node in nodes.items():
            if node is not None:
                counts[role] = int(np.count_nonzero(slicer.util.arrayFromVolume(node)))
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
        trab_label=100,
        cort_label=127,
        output_name="HRpQCT_HOM_material_labels",
    ):
        if bone_segmentation_node is None:
            raise ValueError("Select a bone segmentation labelmap.")

        seg = self._mask_array_from_node(bone_segmentation_node)
        masks = derive_compartment_mask_arrays(
            full=self._mask_array_from_node(full_mask_node),
            trab=self._mask_array_from_node(trab_mask_node),
            cort=self._mask_array_from_node(cort_mask_node),
            output_role="auto",
        )
        if seg.shape != masks["trab"].shape:
            raise ValueError(
                f"Bone segmentation shape {seg.shape} does not match compartment mask shape {masks['trab'].shape}."
            )
        cort_source = (
            "cort_mask"
            if cort_mask_node is not None
            else "derived_from_full_minus_trab"
            if full_mask_node is not None and trab_mask_node is not None
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
        selector.nodeTypes = ["vtkMRMLLabelMapVolumeNode", "vtkMRMLScalarVolumeNode"]
        selector.selectNodeUponCreation = False
        selector.addEnabled = False
        selector.removeEnabled = False
        selector.noneEnabled = True
        selector.setMRMLScene(slicer.mrmlScene)
        return selector

    def _build_ui(self):
        self.messageLabel = qt.QLabel()
        self.messageLabel.wordWrap = True
        self.layout.addWidget(self.messageLabel)

        form = qt.QFormLayout()
        self.layout.addLayout(form)
        self.materialSegSelector = self._labelmap_selector()
        self.materialTrabSelector = self._labelmap_selector()
        self.materialCortSelector = self._labelmap_selector()
        self.materialFullSelector = self._labelmap_selector()
        self._tip(self.materialSegSelector, "Bone segmentation labelmap used to restrict material labels to segmented bone voxels.")
        self._tip(self.materialTrabSelector, "Trabecular ROI mask.")
        self._tip(self.materialCortSelector, "Cortical ROI mask.")
        self._tip(self.materialFullSelector, "Full/periosteal ROI mask.")
        form.addRow("Bone segmentation", self.materialSegSelector)
        form.addRow("Trabecular mask", self.materialTrabSelector)
        form.addRow("Cortical mask", self.materialCortSelector)
        form.addRow("Full mask", self.materialFullSelector)

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
        self.maskASelector = self._labelmap_selector()
        self.maskBSelector = self._labelmap_selector()
        self.maskOperationCombo = qt.QComboBox()
        for label, value in [("Union", "union"), ("Intersection", "intersection"), ("A minus B", "difference"), ("XOR", "xor")]:
            self.maskOperationCombo.addItem(label, value)
        self.maskOperationOutputNameEdit = qt.QLineEdit("HRpQCT_mask_operation")
        self.createMaskOperationButton = qt.QPushButton("Create Mask Operation")
        self.createMaskOperationButton.clicked.connect(self._create_mask_operation)
        operations_form.addRow("Mask A", self.maskASelector)
        operations_form.addRow("Mask B", self.maskBSelector)
        operations_form.addRow("Operation", self.maskOperationCombo)
        operations_form.addRow("Output name", self.maskOperationOutputNameEdit)
        operations_form.addRow(self.createMaskOperationButton)
        self.layout.addWidget(operations_box)

        relabel_box = qt.QGroupBox("Relabel And Validate")
        relabel_form = qt.QFormLayout(relabel_box)
        self.relabelSourceSelector = self._labelmap_selector()
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
        relabel_form.addRow("Source", self.relabelSourceSelector)
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
