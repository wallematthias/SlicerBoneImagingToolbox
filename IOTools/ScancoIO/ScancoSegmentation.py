from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from ScancoIO import ScancoIOVariantFileReader, SEGMENTATION_READER_EXTENSIONS


class ScancoSegmentation(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "ScancoSegmentation"
        parent.categories = []
        parent.hidden = True
        parent.dependencies = ["ScancoIO"]
        parent.contributors = ["Matthias Walle"]
        parent.helpText = "Hidden Scanco native segmentation drag/drop reader."
        parent.acknowledgementText = "Part of the Bone Imaging Toolbox for 3D Slicer."


class ScancoSegmentationFileReader(ScancoIOVariantFileReader):
    def __init__(self, parent):
        super().__init__(
            parent,
            description="ScancoSegmentation",
            file_type="ScancoSegmentation",
            scaling="native",
            load_as="segmentation",
            extensions=SEGMENTATION_READER_EXTENSIONS,
        )
