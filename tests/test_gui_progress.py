"""GUI unit tests for progress indicator and related hooks."""

from __future__ import annotations

from typing import Any, List

import pytest
from PySide6.QtCore import QEvent

from VoxelTree.gui import app as gui_app
from VoxelTree.gui.detail_panel import DetailPanel
from VoxelTree.gui.run_registry import RunRegistry
from VoxelTree.gui.step_node_widget import StepNodeWidget


@pytest.fixture(autouse=True)
def ensure_app():
    # make sure a QApplication exists for widget construction
    gui_app.create_app()
    yield


def test_stepnode_progress_and_icon():
    widget = StepNodeWidget("s1", "S1")

    # initially no progress, status not_run
    assert widget._progress is None
    assert widget._icon() == "-"

    widget.set_status("running")
    assert widget._status == "running"
    # blinking should be active since no progress yet
    assert widget._pulse

    # set some progress; it should clamp and suppress pulse
    widget.set_progress(0.4)
    assert widget._progress == 0.4
    assert not widget._pulse
    assert widget._icon() == "40%"

    # metadata overrides percentage
    widget.set_metadata("5e")
    assert widget._icon() == "5e"

    # clear metadata but keep progress
    widget.set_metadata(None)
    assert widget._icon() == "40%"

    # update progress beyond bounds
    widget.set_progress(1.5)
    assert widget._progress == 1.0
    assert widget._icon() == "100%"

    # moving to success clears progress
    widget.set_status("success")
    assert widget._progress is None
    assert widget._icon() == "✓"

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
    panel._on_progress("foo", None)  # type: ignore[arg-type]
    assert reg.get_metadata("foo", "progress") is None
    # parent callback still invoked with None step progress
    assert parent.calls[-1] == ("p", "foo")


def test_profilerow_refresh_shows_progress():
    # create a dummy registry with a fake progress value
    reg = RunRegistry("x")
    # pick a real step id so the ProfileRow will create a node for it
    real_step = "train_init"
    reg.set_progress(real_step, 0.33)
    from VoxelTree.gui.profile_row import ProfileRow

    row = ProfileRow("x", reg)
    # ensure the node exists and initial progress is shown after refresh
    row.refresh()
    node = row._nodes.get(real_step)
    assert node is not None
    assert node._progress == 0.33
    assert node._icon() == "33%"
