from __future__ import annotations

from pathlib import Path
import sys

import ctk
import qt
import slicer

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)


TOOLBOX_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLBOX_ROOT))

from SlicerBoneImagingToolboxLib.package_status import (  # noqa: E402
    DEFAULT_RUNTIME_PACKAGES,
    PackageStatusRow,
    install_commands,
    package_status_rows,
)
from SlicerBoneImagingToolboxLib.registry import TOOLBOX_DISPLAY_NAME  # noqa: E402
from SlicerBoneImagingToolboxLib.slicer_pip import slicer_pip_install  # noqa: E402
from SlicerBoneImagingToolboxLib.updater import (  # noqa: E402
    ToolboxUpdateCheck,
    check_for_updates,
    short_revision,
    update_toolbox,
)


MODULE_VERSION = "0.1.0"
STATUS_LABELS = {
    "missing": "Missing",
    "update_available": "Update available",
    "current": "Current",
    "current_unknown_latest": "Current; latest unknown",
}


class BoneImagingToolboxSetup(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "Toolbox Setup"
        parent.categories = ["Bone Imaging.Setup"]
        parent.dependencies = []
        parent.contributors = ["Matthias Walle"]
        parent.helpText = (
            f"Central setup and update dashboard for {TOOLBOX_DISPLAY_NAME}.\n"
            f"Module version: {MODULE_VERSION}"
        )
        parent.acknowledgementText = "Built for streamlined Bone Imaging Toolbox installation and updates."


class BoneImagingToolboxSetupLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        super().__init__()
        self._last_toolbox_check: ToolboxUpdateCheck | None = None

    def toolbox_root(self):
        return TOOLBOX_ROOT

    def check_toolbox(self):
        self._last_toolbox_check = check_for_updates(__file__)
        return self._last_toolbox_check

    def update_checked_toolbox(self):
        latest = self._last_toolbox_check.latest_revision if self._last_toolbox_check else None
        return update_toolbox(__file__, latest_revision=latest)

    def package_rows(self, *, check_latest=True):
        latest_versions = None
        if not check_latest:
            latest_versions = {}
        return package_status_rows(DEFAULT_RUNTIME_PACKAGES, latest_versions=latest_versions, timeout=3)

    def install_or_update_package(self, row: PackageStatusRow):
        commands = install_commands(row.spec, installed=row.installed_version is not None)
        for command in commands:
            slicer_pip_install(command)
        for name in list(sys.modules):
            if name == row.spec.import_name or name.startswith(f"{row.spec.import_name}."):
                sys.modules.pop(name, None)
        return " && ".join(commands)


class BoneImagingToolboxSetupWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        super().setup()
        self.logic = BoneImagingToolboxSetupLogic()
        self._last_package_rows = ()

        toolbox_box = ctk.ctkCollapsibleButton()
        toolbox_box.text = "Toolbox"
        self.layout.addWidget(toolbox_box)
        toolbox_layout = qt.QVBoxLayout(toolbox_box)

        self.toolboxStatusLabel = qt.QLabel()
        self.toolboxStatusLabel.wordWrap = True
        toolbox_layout.addWidget(self.toolboxStatusLabel)

        toolbox_buttons = qt.QHBoxLayout()
        self.checkToolboxButton = qt.QPushButton("Check for updates")
        self.updateToolboxButton = qt.QPushButton("Update toolbox")
        self.updateToolboxButton.enabled = False
        self.checkToolboxButton.toolTip = "Check whether the installed toolbox code has an upstream update."
        self.updateToolboxButton.toolTip = "Update this toolbox checkout or manual download when an update is available."
        self.checkToolboxButton.clicked.connect(self._check_toolbox_updates)
        self.updateToolboxButton.clicked.connect(self._update_toolbox)
        toolbox_buttons.addWidget(self.checkToolboxButton)
        toolbox_buttons.addWidget(self.updateToolboxButton)
        toolbox_layout.addLayout(toolbox_buttons)

        packages_box = ctk.ctkCollapsibleButton()
        packages_box.text = "Runtime packages"
        self.layout.addWidget(packages_box)
        packages_layout = qt.QVBoxLayout(packages_box)

        self.packageStatusLabel = qt.QLabel()
        self.packageStatusLabel.wordWrap = True
        packages_layout.addWidget(self.packageStatusLabel)

        self.packageTable = qt.QTableWidget()
        self.packageTable.setColumnCount(5)
        self.packageTable.setHorizontalHeaderLabels(["Package", "Installed", "Latest", "Status", "Action"])
        self.packageTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.packageTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.packageTable.horizontalHeader().setStretchLastSection(True)
        packages_layout.addWidget(self.packageTable)

        package_buttons = qt.QHBoxLayout()
        self.checkPackagesButton = qt.QPushButton("Check package updates")
        self.installRecommendedButton = qt.QPushButton("Install / update needed")
        self.checkPackagesButton.toolTip = "Refresh installed versions and compare each runtime package with PyPI."
        self.installRecommendedButton.toolTip = "Install missing packages and update packages with newer PyPI releases."
        self.checkPackagesButton.clicked.connect(self._refresh_packages)
        self.installRecommendedButton.clicked.connect(self._install_needed_packages)
        package_buttons.addWidget(self.checkPackagesButton)
        package_buttons.addWidget(self.installRecommendedButton)
        packages_layout.addLayout(package_buttons)

        self.logText = qt.QTextEdit()
        self.logText.readOnly = True
        self.logText.minimumHeight = 120
        self.logText.placeholderText = "Setup log"
        self.layout.addWidget(self.logText)

        self.layout.addStretch(1)
        self._set_initial_status()
        qt.QTimer.singleShot(0, self._refresh_packages)

    def _set_initial_status(self):
        self.toolboxStatusLabel.text = f"{TOOLBOX_DISPLAY_NAME} checkout: {self.logic.toolbox_root()}"
        self.packageStatusLabel.text = "Checking installed package versions and latest PyPI releases..."

    def _log(self, message):
        self.logText.append(str(message).rstrip())
        self.logText.ensureCursorVisible()

    def _with_wait_cursor(self, func):
        try:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            return func()
        finally:
            try:
                slicer.app.restoreOverrideCursor()
            except Exception:
                pass

    def _check_toolbox_updates(self):
        try:
            check = self._with_wait_cursor(self.logic.check_toolbox)
        except Exception as exc:
            slicer.util.errorDisplay(f"Toolbox update check failed:\n{exc}")
            self._log(f"[toolbox] update check failed: {exc}")
            return

        installed = short_revision(check.installed_revision)
        latest = short_revision(check.latest_revision)
        self.toolboxStatusLabel.text = (
            f"{TOOLBOX_DISPLAY_NAME}: installed {installed}; latest {latest}. {check.message}"
        )
        self.updateToolboxButton.enabled = bool(check.update_available)
        self._log(f"[toolbox] {check.message}")

    def _update_toolbox(self):
        check = getattr(self.logic, "_last_toolbox_check", None)
        if check is None:
            self._check_toolbox_updates()
            check = getattr(self.logic, "_last_toolbox_check", None)
        if check is None or not check.update_available:
            return

        if check.context.strategy == "git":
            prompt = (
                f"{check.message}\n\n"
                f"Installed folder:\n{check.context.toolbox_root}\n\n"
                "This is a git checkout. Continue with a fast-forward update?"
            )
        else:
            prompt = (
                f"{check.message}\n\n"
                f"Installed folder:\n{check.context.toolbox_root}\n\n"
                "This manual download will be replaced after creating a backup. Continue?"
            )
        if not slicer.util.confirmYesNoDisplay(prompt, windowTitle=f"Update {TOOLBOX_DISPLAY_NAME}"):
            return

        try:
            result = self._with_wait_cursor(self.logic.update_checked_toolbox)
        except Exception as exc:
            slicer.util.errorDisplay(f"Toolbox update failed:\n{exc}")
            self._log(f"[toolbox] update failed: {exc}")
            return

        self._log(f"[toolbox] {result.message}")
        self.updateToolboxButton.enabled = False
        message = result.message
        if result.restart_required:
            message = f"{message}\n\nRestart Slicer to use the updated toolbox."
        slicer.util.infoDisplay(
            message,
            windowTitle=f"{TOOLBOX_DISPLAY_NAME} updated",
        )

    def _refresh_packages(self):
        try:
            rows = self._with_wait_cursor(lambda: self.logic.package_rows(check_latest=True))
        except Exception as exc:
            slicer.util.errorDisplay(f"Package status check failed:\n{exc}")
            self._log(f"[packages] status check failed: {exc}")
            return
        self._last_package_rows = rows
        self._populate_package_table(rows)
        actions = [row for row in rows if row.action]
        self.installRecommendedButton.enabled = bool(actions)
        self.packageStatusLabel.text = (
            f"{len(actions)} package update{'s' if len(actions) != 1 else ''} needed."
            if actions
            else "All checked runtime packages are current or have no available PyPI status."
        )
        self._log("[packages] status refreshed")

    def _populate_package_table(self, rows):
        self.packageTable.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                row.spec.display_name,
                row.installed_version or "Not installed",
                row.latest_version or "Unknown",
                STATUS_LABELS.get(row.status, row.status),
            ]
            for col_index, value in enumerate(values):
                item = qt.QTableWidgetItem(str(value))
                item.setToolTip(row.detail or row.spec.notes)
                self.packageTable.setItem(row_index, col_index, item)

            self.packageTable.setCellWidget(row_index, 4, self._action_button(row))
        self.packageTable.resizeColumnsToContents()

    def _action_button(self, row):
        if not row.action:
            label = qt.QLabel("")
            label.toolTip = row.detail or row.spec.notes
            return label

        label = "Install" if row.action == "install" else "Update"
        button = qt.QPushButton(label)
        button.toolTip = f"{label} {row.spec.display_name} in Slicer Python."
        button.clicked.connect(lambda _checked=False, status_row=row: self._install_package(status_row))
        return button

    def _install_package(self, row):
        action = "install" if row.installed_version is None else "update"
        prompt = (
            f"{row.spec.display_name}\n\n"
            f"Installed: {row.installed_version or 'not installed'}\n"
            f"Latest: {row.latest_version or 'unknown'}\n\n"
            f"Continue with package {action} in Slicer Python?"
        )
        if not slicer.util.confirmYesNoDisplay(prompt, windowTitle=f"{action.title()} runtime package"):
            return False
        try:
            command = self._with_wait_cursor(lambda: self.logic.install_or_update_package(row))
        except Exception as exc:
            slicer.util.errorDisplay(f"Package {action} failed:\n{exc}")
            self._log(f"[packages] {row.spec.package_name} {action} failed: {exc}")
            return False
        self._log(f"[packages] pip install {command}")
        self._refresh_packages()
        return True

    def _install_needed_packages(self):
        rows = [row for row in self._last_package_rows if row.action]
        if not rows:
            self._refresh_packages()
            rows = [row for row in self._last_package_rows if row.action]
        if not rows:
            return
        names = "\n".join(f"- {row.spec.display_name}" for row in rows)
        if not slicer.util.confirmYesNoDisplay(
            f"Install or update these runtime packages in Slicer Python?\n\n{names}",
            windowTitle="Install runtime packages",
        ):
            return
        for row in rows:
            try:
                command = self._with_wait_cursor(lambda status_row=row: self.logic.install_or_update_package(status_row))
            except Exception as exc:
                slicer.util.errorDisplay(f"Package install/update failed:\n{row.spec.display_name}\n\n{exc}")
                self._log(f"[packages] {row.spec.package_name} failed: {exc}")
                break
            self._log(f"[packages] pip install {command}")
        self._refresh_packages()


class BoneImagingToolboxSetupTest(ScriptedLoadableModuleTest):
    def runTest(self):
        self.delayDisplay("BoneImagingToolboxSetup smoke test passed.")
