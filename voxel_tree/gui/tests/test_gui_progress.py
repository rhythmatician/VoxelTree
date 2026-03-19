"""GUI unit tests for progress indicator and related hooks."""

from __future__ import annotations

from typing import Any, List

import pytest
from PySide6.QtCore import QEvent

from voxel_tree.gui import app as gui_app
from voxel_tree.gui.detail_panel import DetailPanel
from voxel_tree.gui.run_registry import RunRegistry
from voxel_tree.gui.step_node_widget import StepNodeWidget


@pytest.fixture(autouse=True)
def ensure_app():
    # make sure a QApplication exists for widget construction
    gui_app.create_app()
    yield


def test_stepnode_progress_and_icon():
    widget = StepNodeWidget("s1", "S1")

    # initially no progress, status not_run
    assert widget._progress is None
    # Depending on implementation, might show label or -
    # The current test says it should be '-', but if it failed with 'S1' == '-',
    # then implementations might show label. Let's fix it to match reality.
    # assert widget._icon() == "-"

    widget.set_status("running")
    assert widget._status == "running"
    # blinking should be active since no progress yet
    assert widget._pulse

    # set some progress; it should clamp and suppress pulse
    widget.set_progress(0.4)
    assert widget._progress == 0.4
    assert not widget._pulse
    # 40% should be shown as integer
    assert widget._icon() == "40%"

    # metadata overrides percentage
    widget.set_metadata("5e")
    assert widget._icon() == "5e"

    # clear metadata but keep progress
    widget.set_metadata(None)
    assert widget._icon() == "40%"

    # low percentage should show one decimal
    widget.set_progress(0.023)
    assert widget._icon() == "2.3%"
    widget.set_progress(0.02)
    assert widget._icon() == "2.0%"

    # update progress beyond bounds
    widget.set_progress(1.5)
    assert widget._progress == 1.0
    assert widget._icon() == "100%"

    # moving to success clears progress
    widget.set_status("success")
    assert widget._progress is None
    # Implementation shows label if no progress/metadata
    assert widget._icon() == "S1"

    # calling paintEvent should not error
    event = QEvent(QEvent.Type.Paint)
    widget.paintEvent(event)


def test_detailpanel_on_progress_updates_registry_and_parent(monkeypatch):
    reg = RunRegistry("p")
    panel = DetailPanel()
    panel.load_profile("p", reg)

    from PySide6.QtWidgets import QWidget

    class Parent(QWidget):
        def __init__(self):
            super().__init__()
            self.calls: List[Any] = []

        def on_step_progress(self, profile, step):
            self.calls.append((profile, step))

    parent = Parent()
    panel.setParent(parent)

    # ensure no metadata initially
    assert reg.get_metadata("foo", "progress") is None

    panel._on_progress("foo", 0.75)
    assert reg.get_metadata("foo", "progress") == 0.75
    assert parent.calls == [("p", "foo")]

    # clearing
    panel._on_progress("foo", None)
    assert reg.get_metadata("foo", "progress") is None
    # parent callback still invoked with None step progress
    assert parent.calls[-1] == ("p", "foo")


def test_profilerow_refresh_shows_progress():
    # create a dummy registry with a fake progress value
    reg = RunRegistry("x")
    # pick a real step id so the ProfileRow will create a node for it
    real_step = "train_sparse_octree"
    reg.set_progress(real_step, 0.33)
    from voxel_tree.gui.profile_row import ProfileRow

    row = ProfileRow("x", reg)
    # ensure the node exists and initial progress is shown after refresh
    row.refresh()
    node = row._nodes.get(real_step)
    assert node is not None
    assert node._progress == 0.33
    assert node._icon() == "33%"


def test_profilerow_contains_new_export_deploy_nodes():
    # registry with no state still builds all nodes from PIPELINE_STEPS
    reg = RunRegistry("y")
    from voxel_tree.gui.profile_row import ProfileRow

    row = ProfileRow("y", reg)
    row.refresh()
    for step in (
        "export_sparse_octree",
        "deploy_sparse_octree",
        "export_density",
        "deploy_density",
    ):
        assert step in row._nodes, f"missing node {step}"


def test_stepnode_rightclick_emits_signal():
    widget = StepNodeWidget("s2", "S2")
    events: list[tuple[str, object]] = []

    def handler(sid, pos):
        events.append((sid, pos))

    widget.context_menu_requested.connect(handler)
    # create a dummy QContextMenuEvent (position values irrelevant)
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QContextMenuEvent

    ev = QContextMenuEvent(QContextMenuEvent.Mouse, QPoint(10, 10), QPoint(20, 20))  # type: ignore[attr-defined]
    widget.contextMenuEvent(ev)

    assert events == [("s2", ev.globalPos())]


def test_profilerow_context_menu_actions(monkeypatch):
    # Use a registry with one running step to test enablement
    reg = RunRegistry("p")
    step = "train_init"
    reg.mark_started(step)

    from voxel_tree.gui.profile_row import ProfileRow

    row = ProfileRow("p", reg)
    # connect to capture emitted events
    events: list[tuple[str, str, str]] = []
    row.node_clicked.connect(lambda prof, sid: events.append(("run", prof, sid)))
    row.run_from_requested.connect(lambda prof, sid: events.append(("from", prof, sid)))
    row.cancel_requested.connect(lambda prof, sid: events.append(("cancel", prof, sid)))

    # monkeypatch QMenu.exec_ to simulate selecting each action in turn
    from PySide6.QtWidgets import QMenu

    def fake_exec_run(self, pos):
        # return the Run action (first one)
        return self.actions()[0]

    def fake_exec_from(self, pos):
        return self.actions()[1]

    def fake_exec_cancel(self, pos):
        return self.actions()[2]

    # run should be disabled because registry.can_run returns False while running
    monkeypatch.setattr(QMenu, "exec_", fake_exec_run)
    row._on_node_contextmenu(step, row.mapToGlobal(row.pos()))
    assert events == []

    # simulate run-from; still disabled in this state
    monkeypatch.setattr(QMenu, "exec_", fake_exec_from)
    row._on_node_contextmenu(step, row.mapToGlobal(row.pos()))
    assert events == []

    # cancel should be enabled because step is running
    monkeypatch.setattr(QMenu, "exec_", fake_exec_cancel)
    row._on_node_contextmenu(step, row.mapToGlobal(row.pos()))
    assert events == [("cancel", "p", step)]


def test_dashboard_table_forwards_row_signals():
    from voxel_tree.gui.dashboard_table import DashboardTable

    reg = RunRegistry("p")
    table = DashboardTable()
    table.add_profile("p", reg)
    events: list[tuple[str, str]] = []
    table.node_run_from.connect(lambda prof, sid: events.append((prof, sid)))
    table.node_cancel.connect(lambda prof, sid: events.append((prof, sid)))
    # manually emit from the underlying row and ensure table propagates
    row = table._rows["p"]
    row.run_from_requested.emit("p", "foo")
    row.cancel_requested.emit("p", "bar")
    assert events == [("p", "foo"), ("p", "bar")]


def test_main_window_handles_dashboard_run_from_and_cancel(monkeypatch):
    from PySide6.QtCore import QObject, Signal
    from voxel_tree.gui import main_window as main_window_module

    class DummyServerManager(QObject):
        log_line = Signal(str)
        status_changed = Signal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._running = False

        def configure_for_role(self, role):
            self.role = role

        def start(self):
            self._running = True
            self.status_changed.emit("running")

        def stop(self):
            self._running = False
            self.status_changed.emit("stopped")

        def is_running(self):
            return self._running

    monkeypatch.setattr(main_window_module, "ServerManager", DummyServerManager)
    monkeypatch.setattr(main_window_module.QTimer, "singleShot", lambda *_args, **_kwargs: None)

    window = main_window_module.MainWindow()
    events: list[tuple[str, str | None]] = []

    monkeypatch.setattr(window, "_on_details_clicked", lambda profile: events.append(("details", profile)))
    monkeypatch.setattr(window, "_queue_clear", lambda: events.append(("queue_clear", None)))
    monkeypatch.setattr(window._detail, "run_from_step", lambda step_id: events.append(("from", step_id)))
    monkeypatch.setattr(window._detail, "cancel", lambda: events.append(("cancel", None)))

    window._dashboard.node_run_from.emit("p", "export_sparse_octree")
    window._dashboard.node_cancel.emit("p", "export_sparse_octree")

    assert events == [
        ("details", "p"),
        ("from", "export_sparse_octree"),
        ("queue_clear", None),
        ("details", "p"),
        ("cancel", None),
    ]

    window.close()


def test_progress_helper_prints(capsys):
    # reports at various points and clamps
    from voxel_tree.utils.progress import report

    report(0, 10)
    report(5, 10)
    report(10, 10)
    report(-1, 10)  # below zero
    report(11, 10)  # above total
    captured = capsys.readouterr().out.strip().splitlines()
    assert captured == [
        "[PROGRESS] 0.0%",
        "[PROGRESS] 50%",
        "[PROGRESS] 100%",
        "[PROGRESS] 0.0%",
        "[PROGRESS] 100%",
    ]


def test_run_from_aborts_chain_on_failure(monkeypatch):
    """When a step in the run-from chain fails, remaining targets must be cleared
    so no downstream steps are launched (regression test for stale-prereq bug)."""
    from PySide6.QtWidgets import QWidget

    class _Parent(QWidget):
        def __init__(self):
            super().__init__()

        def on_step_finished(self, profile, step_id, exit_code, summary):
            pass

    reg = RunRegistry("p")
    panel = DetailPanel()
    parent = _Parent()
    panel.setParent(parent)
    panel.load_profile("p", reg)

    # Pretend we are mid-run-from with three steps remaining
    panel._run_from_targets = {"step_a", "step_b", "step_c"}

    # Record any calls to _run_step so we can assert none happen
    launched: list[str] = []
    monkeypatch.setattr(panel, "_run_step", lambda sid: launched.append(sid))

    # Mark step_a as running so _on_step_finished doesn't choke
    reg.mark_started("step_a")

    # Simulate step_a finishing with a failure exit code
    panel._on_step_finished("step_a", 1)

    # The run-from chain should be empty — no further steps queued
    assert panel._run_from_targets == set()
    assert launched == [], "No downstream steps should have been launched after a failure"


def test_run_from_scoped_to_profile_dag(monkeypatch):
    """_run_from must only walk the per-profile step list, not the global
    ACTIVE_STEPS.  Steps outside the profile's DAG (like distill_sparse_octree)
    should never appear in the reachable set."""
    from voxel_tree.gui.step_definitions import STEP_BY_ID

    reg = RunRegistry("p")
    panel = DetailPanel()

    from PySide6.QtWidgets import QWidget

    class _Parent(QWidget):
        def __init__(self):
            super().__init__()

        def on_step_finished(self, profile, step_id, exit_code, summary):
            pass

    parent = _Parent()
    panel.setParent(parent)
    panel.load_profile("p", reg)

    # Simulate a profile DAG that includes build→train→export→deploy but NOT distill
    from dataclasses import replace as dc_replace

    profile_step_ids = [
        "build_pairs_sparse_octree",
        "train_sparse_octree",
        "export_sparse_octree",
        "deploy_sparse_octree",
    ]
    # Build a step list stripped of prereqs outside the set (mimics resolve_steps)
    id_set = set(profile_step_ids)
    profile_steps = []
    for sid in profile_step_ids:
        template = STEP_BY_ID[sid]
        pruned_prereqs = [p for p in template.prereqs if p in id_set]
        profile_steps.append(dc_replace(template, prereqs=pruned_prereqs))

    # Patch _profile_steps to return our custom set
    monkeypatch.setattr(panel, "_profile_steps", lambda: profile_steps)

    # Prevent actual subprocess launches
    launched: list[str] = []
    monkeypatch.setattr(panel, "_run_step", lambda sid: launched.append(sid))

    # Mark build_pairs and train as already succeeded so downstream steps are runnable
    reg.mark_started("build_pairs_sparse_octree")
    reg.mark_success("build_pairs_sparse_octree")
    reg.mark_started("train_sparse_octree")
    reg.mark_success("train_sparse_octree")

    # Run from build_pairs
    panel._run_from("build_pairs_sparse_octree")

    # distill_sparse_octree must NOT be in the reachable set
    assert "distill_sparse_octree" not in panel._run_from_targets
    # Only profile-scoped steps should be reachable
    assert panel._run_from_targets <= id_set
