"""profile_editor.py — Dialog to create or edit a run profile YAML."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Determine the project root by walking up from this module until we find
# a repository marker (pyproject.toml or .git). This works correctly both when
# running from the source tree and when installed in editable/packaged mode.
#
# This is important because the code lives under ``VoxelTree/VoxelTree/`` when
# checking out the repo, but the repo root (where ``profiles/`` and ``runs/``
# live) is one level above that.

def _find_project_root(start: Path) -> Path:
    for ancestor in [start] + list(start.parents):
        if (ancestor / "pyproject.toml").exists() or (ancestor / ".git").exists():
            return ancestor
    return start


_PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
_PROFILES_DIR = _PROJECT_ROOT / "profiles"
_RUNS_DIR = _PROJECT_ROOT / "runs"

# Default profile structure — used when creating a new profile
_DEFAULT_PROFILE: dict = {
    "name": "new_profile",
    "description": "",
    "world": {
        "seed": 12345,
        "radius": 512,
        "save_name": "New World",
    },
    "rcon": {
        "host": "localhost",
        "port": 25575,
        "password": "",
        "timeout": 300,
    },
    "data": {
        "voxy_dir": "../LODiffusion/run/saves",
        "data_dir": "data/voxy_octree",
        "max_sections": 1000,
        "min_solid": 0.02,
        "val_split": 0.1,
        "noise_dump_dir": "tools/fabric-server/runtime/noise_dumps",
    },
    "train": {
        "output_dir": "models/new_profile",
        "max_samples": 5000,
        "epochs": 20,
        "batch_size": 4,
        "lr": 0.0001,
        "device": "auto",
    },
    "export": {
        "output_dir": "production/new_profile",
    },
    "deploy": {
        "target_dir": "../LODiffusion/run/config/lodiffusion",
    },
}


def load_profile(profile_name: str) -> dict:
    """Load a profile from profiles/<name>.yaml.  Returns empty dict if not found."""
    path = _PROFILES_DIR / f"{profile_name}.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def save_profile(data: dict) -> None:
    """Save profile dict to profiles/<name>.yaml."""
    _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    name = data.get("name", "unnamed")
    path = _PROFILES_DIR / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def list_profiles() -> list[str]:
    """Return sorted list of profile names (stems of *.yaml in profiles/)."""
    _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.stem for p in _PROFILES_DIR.glob("*.yaml"))


def delete_profile(profile_name: str) -> bool:
    """Delete a profile YAML file. Returns True if successful, False otherwise."""
    path = _PROFILES_DIR / f"{profile_name}.yaml"
    if path.exists():
        path.unlink()
        return True
    return False


def delete_profile_data(paths: list[Path]) -> None:
    """Delete each path — unlinks files, rmtrees directories.  Silently skips missing."""
    for p in paths:
        if not p.exists():
            continue
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


class ProfileDeleteDialog(QDialog):
    """Confirmation dialog for deleting a profile and its per-profile artifacts.

    Shows checkboxes for each artifact that exists on disk.  The caller
    can query ``selected_paths()`` after ``exec()`` returns ``Accepted``.
    """

    def __init__(
        self,
        profile_name: str,
        profile_data: dict,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Delete Profile — {profile_name}")
        self.setMinimumWidth(460)
        self.setStyleSheet("background: #1e1e1e; color: #cccccc;")

        self._checkboxes: list[tuple[QCheckBox, Path]] = []

        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # ── Title ──
        title = QLabel(f"Delete <b>{profile_name}</b>")
        title.setStyleSheet("font-size: 14px; color: #ff9999;")
        root.addWidget(title)

        # ── Artifact checkboxes ──
        items_box = QGroupBox("Items to delete")
        items_box.setStyleSheet(
            "QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 8px;"
            " padding-top: 8px; color: #aaaaaa; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        items_layout = QVBoxLayout(items_box)
        items_layout.setSpacing(6)

        def _add_item(label: str, path: Path, exists_required: bool = True) -> None:
            if exists_required and not path.exists():
                return
            rel = path.relative_to(_PROJECT_ROOT) if path.is_relative_to(_PROJECT_ROOT) else path
            chk = QCheckBox(f"{label}\n  {rel}")
            chk.setChecked(True)
            chk.setStyleSheet("color: #cccccc; font-size: 11px;")
            items_layout.addWidget(chk)
            self._checkboxes.append((chk, path))

        # Always-present items
        _add_item("Profile YAML", _PROFILES_DIR / f"{profile_name}.yaml", exists_required=False)
        _add_item("Run state", _RUNS_DIR / profile_name, exists_required=False)

        # Per-profile directories derived from the profile YAML
        train_out = profile_data.get("train", {}).get("output_dir", "")
        if train_out:
            _add_item("Trained model", (_PROJECT_ROOT / train_out).resolve())

        export_out = profile_data.get("export", {}).get("output_dir", "")
        if export_out:
            _add_item("Exported files", (_PROJECT_ROOT / export_out).resolve())

        root.addWidget(items_box)

        # ── Shared-data info note ──
        shared_data_dir = profile_data.get("data", {}).get("data_dir", "data/voxy_octree")
        note = QLabel(
            f"ℹ  <b>Shared data is never deleted.</b><br>"
            f"<span style='color:#888888'>{shared_data_dir} (extract/heights/pairs outputs) "
            f"is used by all profiles and will not be touched.</span>"
        )
        note.setWordWrap(True)
        note.setTextFormat(Qt.TextFormat.RichText)
        note.setStyleSheet(
            "background: #252525; border: 1px solid #444; border-radius: 4px;"
            " padding: 8px; color: #aaaaaa; font-size: 11px;"
        )
        root.addWidget(note)

        # ── Buttons ──
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.setStyleSheet(
            "QPushButton { background: #3a3a3a; color: #cccccc; border: 1px solid #555;"
            " border-radius: 4px; padding: 5px 10px; }"
            "QPushButton:hover { background: #4a4a4a; }"
        )
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        btn_row.addSpacing(8)

        delete_btn = QPushButton("Delete Selected")
        delete_btn.setFixedWidth(120)
        delete_btn.setStyleSheet(
            "QPushButton { background: #6a2a2a; color: #ffaaaa; border: 1px solid #aa4a4a;"
            " border-radius: 4px; padding: 5px 10px; font-weight: bold; }"
            "QPushButton:hover { background: #8a3a3a; }"
        )
        delete_btn.clicked.connect(self.accept)
        btn_row.addWidget(delete_btn)

        root.addLayout(btn_row)

    def selected_paths(self) -> list[Path]:
        """Return the paths whose checkboxes are checked."""
        return [path for chk, path in self._checkboxes if chk.isChecked()]


class ProfileEditorDialog(QDialog):
    """Modal dialog for creating or editing a YAML run profile."""

    def __init__(self, profile_name: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self._is_new = profile_name is None
        self.setWindowTitle("New Profile" if self._is_new else f"Edit Profile — {profile_name}")
        self.resize(480, 560)
        self.setStyleSheet(
            "QDialog { background: #1e1e1e; color: #cccccc; }"
            "QGroupBox { color: #8899bb; font-weight: bold; border: 1px solid #333;"
            " border-radius: 4px; margin-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
            "QLineEdit, QSpinBox, QDoubleSpinBox { background: #2a2a2a; color: #ccc;"
            " border: 1px solid #444; border-radius: 3px; padding: 3px; }"
            "QLabel { color: #aaa; }"
        )

        # Load existing or start from defaults
        if profile_name:
            self._data = load_profile(profile_name)
            if not self._data:
                self._data = dict(_DEFAULT_PROFILE)
        else:
            import copy

            self._data = copy.deepcopy(_DEFAULT_PROFILE)

        self._fields: dict[str, QWidget] = {}
        # Per-profile DAG (None = user has not edited it; kept in sync with _data)
        from VoxelTree.gui.dag_definition import ProfileDag  # late to avoid circular

        self._dag: ProfileDag | None = ProfileDag.from_profile_dict(self._data)
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        inner = QWidget()
        inner.setStyleSheet("background: #1e1e1e;")
        form_layout = QVBoxLayout(inner)
        form_layout.setSpacing(10)
        form_layout.setContentsMargins(10, 10, 10, 10)

        # ── Meta ──
        meta_box = self._group("Profile")
        meta_form = QFormLayout(meta_box)
        self._add_str(meta_form, "name", "Name", self._data.get("name", ""))
        self._add_str(meta_form, "description", "Description", self._data.get("description", ""))
        form_layout.addWidget(meta_box)

        # ── World ──
        world = self._data.get("world", {})
        w_box = self._group("World")
        w_form = QFormLayout(w_box)
        self._add_int(w_form, "world.seed", "Seed", world.get("seed", 12345), 0, 2**31 - 1)
        self._add_int(
            w_form, "world.radius", "Radius (blocks)", world.get("radius", 512), 64, 16384
        )
        self._add_str(w_form, "world.save_name", "Save Name", world.get("save_name", "New World"))
        form_layout.addWidget(w_box)

        # ── RCON ──
        rcon = self._data.get("rcon", {})
        r_box = self._group("RCON")
        r_form = QFormLayout(r_box)
        self._add_str(r_form, "rcon.host", "Host", rcon.get("host", "localhost"))
        self._add_int(r_form, "rcon.port", "Port", rcon.get("port", 25575), 1, 65535)
        self._add_str(r_form, "rcon.password", "Password", rcon.get("password", ""))
        self._add_int(r_form, "rcon.timeout", "Timeout (s)", rcon.get("timeout", 300), 10, 3600)
        form_layout.addWidget(r_box)

        # ── Data ──
        data = self._data.get("data", {})
        d_box = self._group("Data")
        d_form = QFormLayout(d_box)
        self._add_str(
            d_form, "data.voxy_dir", "Voxy Dir", data.get("voxy_dir", "../LODiffusion/run/saves")
        )
        self._add_str(d_form, "data.data_dir", "Data Dir", data.get("data_dir", "data/voxy_octree"))
        self._add_int_opt(d_form, "data.max_sections", "Max Sections", data.get("max_sections"))
        self._add_float(
            d_form, "data.min_solid", "Min Solid Frac", data.get("min_solid", 0.02), 0.0, 1.0
        )
        self._add_float(d_form, "data.val_split", "Val Split", data.get("val_split", 0.1), 0.0, 0.5)
        form_layout.addWidget(d_box)

        # ── Train ──
        train = self._data.get("train", {})
        t_box = self._group("Training")
        t_form = QFormLayout(t_box)
        self._add_str(
            t_form, "train.output_dir", "Output Dir", train.get("output_dir", "models/new_profile")
        )
        self._add_int_opt(t_form, "train.max_samples", "Max Samples", train.get("max_samples"))
        self._add_int(t_form, "train.epochs", "Epochs", train.get("epochs", 20), 1, 10000)
        self._add_int(t_form, "train.batch_size", "Batch Size", train.get("batch_size", 4), 1, 512)
        self._add_float(
            t_form, "train.lr", "Learning Rate", train.get("lr", 0.0001), 1e-6, 0.1, decimals=6
        )
        self._add_str(t_form, "train.device", "Device", train.get("device", "auto"))
        form_layout.addWidget(t_box)

        # ── Export ──
        export = self._data.get("export", {})
        e_box = self._group("Export")
        e_form = QFormLayout(e_box)
        self._add_str(
            e_form,
            "export.output_dir",
            "Output Dir",
            export.get("output_dir", "production/new_profile"),
        )
        form_layout.addWidget(e_box)

        # ── Deploy ──
        deploy = self._data.get("deploy", {})
        dep_box = self._group("Deploy")
        dep_form = QFormLayout(dep_box)
        self._add_str(
            dep_form,
            "deploy.target_dir",
            "Target Dir",
            deploy.get("target_dir", "../LODiffusion/run/config/lodiffusion"),
        )
        form_layout.addWidget(dep_box)

        # ── Pipeline DAG ──
        dag_box = self._group("Pipeline DAG")
        dag_v = QVBoxLayout(dag_box)
        dag_v.setSpacing(6)
        self._dag_summary_label = QLabel()
        self._dag_summary_label.setStyleSheet("color: #7799bb; font-size: 10px; padding: 2px 0;")
        self._dag_summary_label.setWordWrap(True)
        dag_v.addWidget(self._dag_summary_label)
        edit_dag_btn = QPushButton("Edit Pipeline DAG…")
        edit_dag_btn.setFixedWidth(160)
        edit_dag_btn.setStyleSheet(
            "QPushButton { background: #2a3d5a; color: #9abfdd; border: 1px solid #4a7abf;"
            " border-radius: 4px; padding: 4px 10px; }"
            "QPushButton:hover { background: #3a5a8a; }"
        )
        edit_dag_btn.clicked.connect(self._on_edit_dag)
        dag_v.addWidget(edit_dag_btn)
        form_layout.addWidget(dag_box)
        self._update_dag_summary()

        form_layout.addStretch()

        scroll.setWidget(inner)
        root.addWidget(scroll, stretch=1)

        # ── Buttons ──
        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bb.setStyleSheet(
            "QPushButton { background: #2a3a2a; color: #ccc; border: 1px solid #4a8a4a;"
            " border-radius: 4px; padding: 4px 16px; }"
            "QPushButton:hover { background: #3a5a3a; }"
        )
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    # ------------------------------------------------------------------
    # Field helpers
    # ------------------------------------------------------------------

    def _group(self, title: str) -> QGroupBox:
        box = QGroupBox(title)
        return box

    def _add_str(self, form: QFormLayout, key: str, label: str, value: str) -> None:
        w = QLineEdit(str(value))
        form.addRow(label, w)
        self._fields[key] = w

    def _add_int(
        self, form: QFormLayout, key: str, label: str, value: int, mn: int = 0, mx: int = 999999
    ) -> None:
        w = QSpinBox()
        w.setRange(mn, mx)
        w.setValue(int(value))
        form.addRow(label, w)
        self._fields[key] = w

    def _add_int_opt(self, form: QFormLayout, key: str, label: str, value: int | None) -> None:
        """Integer with an 'unlimited' checkbox."""
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        h = QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        spin = QSpinBox()
        spin.setRange(0, 10_000_000)
        spin.setValue(int(value) if value else 0)
        h.addWidget(spin)

        chk = QCheckBox("unlimited")
        chk.setChecked(value is None)
        chk.setStyleSheet("color: #aaa;")
        spin.setEnabled(value is not None)
        chk.toggled.connect(lambda checked: spin.setDisabled(checked))
        h.addWidget(chk)

        form.addRow(label, container)
        self._fields[key] = container
        container._spin = spin  # type: ignore[attr-defined]
        container._chk = chk  # type: ignore[attr-defined]

    def _add_float(
        self,
        form: QFormLayout,
        key: str,
        label: str,
        value: float,
        mn: float = 0.0,
        mx: float = 1.0,
        decimals: int = 4,
    ) -> None:
        w = QDoubleSpinBox()
        w.setDecimals(decimals)
        w.setRange(mn, mx)
        w.setSingleStep(10 ** (-decimals + 1))
        w.setValue(float(value))
        form.addRow(label, w)
        self._fields[key] = w

    # ------------------------------------------------------------------
    # Accept / collect values
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        name = self._get_value("name")
        if not name or not str(name).strip():
            QMessageBox.warning(self, "Validation", "Profile name cannot be empty.")
            return
        self._collect()
        # Persist (or clear) per-profile DAG
        if self._dag is not None and not self._dag.is_empty:
            self._data["dag"] = self._dag.to_dag_dict()
        else:
            self._data.pop("dag", None)
        try:
            save_profile(self._data)
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        self.accept()

    def _get_value(self, key: str):
        w = self._fields.get(key)
        if isinstance(w, QLineEdit):
            return w.text().strip()
        if isinstance(w, QSpinBox):
            return w.value()
        if isinstance(w, QDoubleSpinBox):
            return w.value()
        return None

    def _collect(self) -> None:
        """Write widget values back into self._data."""

        def _set(data: dict, dotted_key: str, value) -> None:
            parts = dotted_key.split(".")
            for part in parts[:-1]:
                data = data.setdefault(part, {})
            data[parts[-1]] = value

        for key, widget in self._fields.items():
            if isinstance(widget, QLineEdit):
                _set(self._data, key, widget.text().strip())
            elif isinstance(widget, QSpinBox):
                _set(self._data, key, widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                _set(self._data, key, widget.value())
            elif hasattr(widget, "_spin") and hasattr(widget, "_chk"):
                # Optional int widget
                if widget._chk.isChecked():  # type: ignore[attr-defined]
                    _set(self._data, key, None)
                else:
                    _set(self._data, key, widget._spin.value())  # type: ignore[attr-defined]

    def profile_name(self) -> str:
        return str(self._data.get("name", ""))

    # ------------------------------------------------------------------
    # DAG helpers
    # ------------------------------------------------------------------

    def _on_edit_dag(self) -> None:
        from VoxelTree.gui.dag_definition import ProfileDag
        from VoxelTree.gui.dag_editor_dialog import DagEditorDialog

        # Use current edited dag, or build a default one if not yet set
        current_dag = (
            self._dag
            if (self._dag is not None and not self._dag.is_empty)
            else ProfileDag.default()
        )
        dlg = DagEditorDialog(
            profile_name=str(self._data.get("name", "?")),
            dag=current_dag,
            parent=self,
        )
        if dlg.exec():
            self._dag = dlg.result_dag()
            self._update_dag_summary()

    def _update_dag_summary(self) -> None:
        dag = self._dag
        if dag is None or dag.is_empty:
            self._dag_summary_label.setText("Default — all PIPELINE_STEPS")
            return
        try:
            steps = dag.resolve_steps()
            labels = "  →  ".join(s.label for s in steps)
            self._dag_summary_label.setText(f"{len(steps)} steps:  {labels}")
        except Exception as exc:
            self._dag_summary_label.setText(f"⚠  {exc}")
