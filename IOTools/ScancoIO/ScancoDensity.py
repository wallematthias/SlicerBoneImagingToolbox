from slicer.ScriptedLoadableModule import ScriptedLoadableModule

from ScancoIO import ScancoIOVariantFileReader, VOLUME_READER_EXTENSIONS


class ScancoDensity(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "ScancoDensity"
        parent.categories = []
        parent.hidden = True
        parent.dependencies = ["ScancoIO"]
        parent.contributors = ["Matthias Walle"]
        parent.helpText = "Hidden Scanco density volume drag/drop reader."
        parent.acknowledgementText = "Author: Matthias Walle. Part of the Bone Imaging Toolbox for 3D Slicer."


class ScancoDensityFileReader(ScancoIOVariantFileReader):
    def __init__(self, parent):
        super().__init__(
            parent,
            description="ScancoDensity",
            file_type="ScancoDensity",
            scaling="density",
            load_as="volume",
            extensions=VOLUME_READER_EXTENSIONS,
        )
