from __future__ import annotations

from pathlib import Path

import numpy as np
import slicer
import vtk


def node_array_shape(node):
    if node is None:
        return None
    try:
        return tuple(int(value) for value in slicer.util.arrayFromVolume(node).shape)
    except Exception:
        return None


def node_storage_file(node):
    if node is None:
        return None
    try:
        storage = node.GetStorageNode()
        filename = storage.GetFileName() if storage is not None else ""
    except Exception:
        return None
    if not filename:
        return None
    path = Path(str(filename)).expanduser()
    return str(path) if path.is_file() else None


def volume_ijk_to_ras_array(volume_node):
    matrix = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(matrix)
    return np.asarray(
        [[float(matrix.GetElement(row, column)) for column in range(4)] for row in range(4)],
        dtype=float,
    )
