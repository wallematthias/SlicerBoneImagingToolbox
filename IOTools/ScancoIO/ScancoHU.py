from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from ScancoIO import ScancoIOVariantFileReader, VOLUME_READER_EXTENSIONS


class ScancoHU(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "ScancoHU"
        parent.categories = []
        parent.hidden = True
        parent.dependencies = ["ScancoIO"]
        parent.contributors = ["Matthias Walle"]
        parent.helpText = "Hidden Scanco HU volume drag/drop reader."
        parent.acknowledgementText = "Author: Matthias Walle. Part of the Bone Imaging Toolbox for 3D Slicer."


class ScancoHUFileReader(ScancoIOVariantFileReader):
    def __init__(self, parent):
        super().__init__(
            parent,
            description="ScancoHU",
            file_type="ScancoHU",
            scaling="hu",
            load_as="volume",
            extensions=VOLUME_READER_EXTENSIONS,
        )
