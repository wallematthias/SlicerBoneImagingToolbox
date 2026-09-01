from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from ScancoIO import ScancoIOVariantFileReader, VOLUME_READER_EXTENSIONS


class ScancoVolume(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "ScancoVolume"
        parent.categories = []
        parent.hidden = True
        parent.dependencies = ["ScancoIO"]
        parent.contributors = ["Matthias Walle"]
        parent.helpText = "Hidden Scanco native volume drag/drop reader."
        parent.acknowledgementText = "Author: Matthias Walle. Part of the Bone Imaging Toolbox for 3D Slicer."


class ScancoVolumeFileReader(ScancoIOVariantFileReader):
    def __init__(self, parent):
        super().__init__(
            parent,
            description="ScancoVolume",
            file_type="ScancoVolume",
            scaling="native",
            load_as="volume",
            extensions=VOLUME_READER_EXTENSIONS,
        )
