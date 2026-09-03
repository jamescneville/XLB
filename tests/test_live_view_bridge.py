"""Tests for the filesystem control bridge between LiveView and the viewer.

The solver and the viewer usually run in different OS images (container vs
host), so their only contract is three files in a shared directory. Both halves
are exercised here against a real temp directory: ``LiveView._write_status`` and
``_poll_camera`` from the solver side, ``Viewer._write_camera`` from the tool
side. Neither GPU nor GUI is needed -- the methods are called unbound against
duck-typed stubs, so nothing constructs a Tk window or a Warp context.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import threading

import pytest

wp = pytest.importorskip("warp")

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lv():
    return _load("xlb_live_view_bridge_under_test", pathlib.Path("xlb") / "utils" / "live_view.py")


@pytest.fixture(scope="module")
def viewer_mod():
    return _load("xlb_live_view_viewer_under_test", pathlib.Path("tools") / "live_view_viewer.py")


class _FakeDisplay:
    index = 7
    completed = 6
    dropped = 2


def _solver_stub(lv, tmp_path):
    """Duck-typed stand-in exposing just what the bridge methods touch."""

    class Stub:
        pass

    s = Stub()
    s._camera_path = str(tmp_path / "camera.json")
    s._status_path = str(tmp_path / "status.json")
    s._camera_mtime = None
    s._recalibrate_seq = 0
    s._render_ms = 12.5
    s._draw_ms = 2.6
    s._data_ms = 240.0
    s._drain_ms = 180.0
    s.q_iso = 3.0e-3
    s.width = 1280
    s.height = 720
    s.rshape = (408, 376, 168)
    s.verbose = False
    s.display = _FakeDisplay()
    s.state = lv._ViewerState(lv.Camera(target=[1.0, 2.0, 3.0], distance=50.0, azimuth=35.0, elevation=20.0))
    return s


def _viewer_stub(viewer_mod, tmp_path, **overrides):
    class Stub:
        pass

    v = Stub()
    v.camera_path = tmp_path / "camera.json"
    v.camera_dirty = True
    v.last_camera_write = 0.0
    v.azimuth = 120.0
    v.elevation = -15.0
    v.distance = 33.0
    v.fov = 45.0
    v.iso_scale = 2.0
    v.recalibrate_seq = 0
    for k, val in overrides.items():
        setattr(v, k, val)
    return v


def test_status_is_written_atomically_and_parses(lv, tmp_path):
    s = _solver_stub(lv, tmp_path)
    lv.LiveView._write_status(s, step=4200, mlups=987.6, iso_scale=1.5)

    status = json.loads((tmp_path / "status.json").read_text())

    assert status["step"] == 4200
    assert status["mlups"] == pytest.approx(987.6)
    assert status["frame"] == 6, "frame index should be the last one queued, not the next"
    assert status["frame_file"] == "live_000006.png"
    assert status["dropped"] == 2
    assert status["q_iso"] == pytest.approx(3.0e-3 * 1.5), "q_iso should be reported post-scale"
    assert status["render_shape"] == [408, 376, 168]
    assert status["azimuth"] == pytest.approx(35.0)

    # No temp file left behind, so a viewer globbing the directory sees only real files.
    assert not list(tmp_path.glob("*.tmp"))


def test_viewer_camera_write_is_read_back_by_solver(lv, viewer_mod, tmp_path):
    """The actual contract: what the viewer writes is what the solver applies."""
    s = _solver_stub(lv, tmp_path)
    v = _viewer_stub(viewer_mod, tmp_path)

    viewer_mod.Viewer._write_camera(v)
    assert not v.camera_dirty, "a successful write should clear the dirty flag"

    assert lv.LiveView._poll_camera(s) is True

    camera = s.state.camera
    assert camera.azimuth == pytest.approx(120.0)
    assert camera.elevation == pytest.approx(-15.0)
    assert camera.distance == pytest.approx(33.0)
    assert camera.fov == pytest.approx(45.0)
    assert s.state.iso_scale == pytest.approx(2.0)

    # Unchanged file must not be re-read; that stat-only path runs every frame.
    assert lv.LiveView._poll_camera(s) is False


def test_recalibrate_uses_a_sequence_not_a_flag(lv, viewer_mod, tmp_path):
    """The viewer cannot observe when a flag was consumed, so it counts up.

    A boolean would either fire once and stick, or need the solver to write back
    into the viewer's file. The sequence number makes each press distinct.
    """
    s = _solver_stub(lv, tmp_path)
    v = _viewer_stub(viewer_mod, tmp_path, recalibrate_seq=0)

    # Baseline: seq 0 matches the solver's initial value, so no recalibration.
    viewer_mod.Viewer._write_camera(v)
    lv.LiveView._poll_camera(s)
    assert s.state.recalibrate is False

    # First press.
    s.state.recalibrate = False
    v.recalibrate_seq = 1
    v.camera_dirty = True
    v.last_camera_write = 0.0
    viewer_mod.Viewer._write_camera(v)
    os.utime(s._camera_path, (1_000_000, 1_000_000))
    lv.LiveView._poll_camera(s)
    assert s.state.recalibrate is True, "a new sequence number must request recalibration"

    # Same sequence again must NOT re-trigger.
    s.state.recalibrate = False
    v.camera_dirty = True
    v.last_camera_write = 0.0
    viewer_mod.Viewer._write_camera(v)
    os.utime(s._camera_path, (2_000_000, 2_000_000))
    lv.LiveView._poll_camera(s)
    assert s.state.recalibrate is False, "an unchanged sequence must not re-trigger"


def test_poll_camera_survives_a_torn_file(lv, tmp_path):
    """A bind mount need not make os.replace visible atomically to the reader.

    A parse failure must leave the previous camera untouched and stay retryable
    rather than latching the bad mtime and ignoring the file forever.
    """
    s = _solver_stub(lv, tmp_path)
    before = s.state.camera.azimuth

    pathlib.Path(s._camera_path).write_text('{"azimuth": 90.0, "eleva')
    assert lv.LiveView._poll_camera(s) is False
    assert s.state.camera.azimuth == pytest.approx(before), "a torn read must not move the camera"
    assert s._camera_mtime is None, "mtime must not latch, or a later good write is ignored"

    # A subsequent complete write is picked up.
    pathlib.Path(s._camera_path).write_text(json.dumps({"azimuth": 90.0}))
    assert lv.LiveView._poll_camera(s) is True
    assert s.state.camera.azimuth == pytest.approx(90.0)


def test_poll_camera_is_a_noop_without_a_bridge(lv, tmp_path):
    """With a real window there is no bridge; the hot path must not touch disk."""
    s = _solver_stub(lv, tmp_path)
    s._camera_path = None
    assert lv.LiveView._poll_camera(s) is False

    s._status_path = None
    lv.LiveView._write_status(s, step=1, mlups=0.0, iso_scale=1.0)
    assert not list(tmp_path.iterdir())


def test_poll_camera_ignores_a_missing_file(lv, tmp_path):
    s = _solver_stub(lv, tmp_path)
    assert not os.path.exists(s._camera_path)
    assert lv.LiveView._poll_camera(s) is False


def test_closed_request_propagates(lv, tmp_path):
    """Closing the viewer with a shutdown request must stop the render loop."""
    s = _solver_stub(lv, tmp_path)
    pathlib.Path(s._camera_path).write_text(json.dumps({"closed": True}))

    lv.LiveView._poll_camera(s)
    assert s.state.closed is True


def test_viewer_throttles_camera_writes(lv, viewer_mod, tmp_path):
    """Each write crosses a bind mount; the solver only reads once per frame."""
    v = _viewer_stub(viewer_mod, tmp_path)

    viewer_mod.Viewer._write_camera(v)
    first = (tmp_path / "camera.json").read_text()

    # Immediately dirty again: the throttle should suppress this write.
    v.camera_dirty = True
    v.azimuth = 999.0
    viewer_mod.Viewer._write_camera(v)
    assert (tmp_path / "camera.json").read_text() == first
    assert v.camera_dirty is True, "a suppressed write must stay pending"

    # Once the interval has elapsed it goes through.
    v.last_camera_write = 0.0
    viewer_mod.Viewer._write_camera(v)
    assert json.loads((tmp_path / "camera.json").read_text())["azimuth"] == pytest.approx(999.0)


def test_find_frame_dir_picks_the_most_recent(viewer_mod, tmp_path):
    old = tmp_path / "run_a" / "live_view"
    new = tmp_path / "run_b" / "live_view"
    for d in (old, new):
        d.mkdir(parents=True)

    (old / "status.json").write_text("{}")
    (new / "status.json").write_text("{}")
    os.utime(old / "status.json", (1_000_000, 1_000_000))
    os.utime(new / "status.json", (2_000_000, 2_000_000))

    assert viewer_mod.find_frame_dir(tmp_path) == new

    # A directory with frames but no status.json still counts, dated by its frames.
    assert viewer_mod.find_frame_dir(tmp_path / "run_a") == old


def test_find_frame_dir_returns_none_when_empty(viewer_mod, tmp_path):
    (tmp_path / "live_view").mkdir()
    assert viewer_mod.find_frame_dir(tmp_path) is None, "an empty live_view dir is not a candidate"


def test_viewer_state_lock_is_reentrant_safe_for_poll(lv, tmp_path):
    """_poll_camera takes the state lock; make sure it is released on every path."""
    s = _solver_stub(lv, tmp_path)
    pathlib.Path(s._camera_path).write_text(json.dumps({"azimuth": 10.0}))
    lv.LiveView._poll_camera(s)

    acquired = s.state.lock.acquire(timeout=1.0)
    assert acquired, "state lock was left held by _poll_camera"
    s.state.lock.release()

    # And on the early-return paths too.
    s._camera_path = None
    lv.LiveView._poll_camera(s)
    assert s.state.lock.acquire(timeout=1.0)
    s.state.lock.release()

    assert isinstance(s.state.lock, type(threading.Lock()))


def test_bc_solid_constant_matches_cell_type(lv):
    """live_view mirrors BC_SOLID so it can be imported by path; keep them equal.

    Only BC_SOLID means "not fluid". Every other nonzero bc_mask value is a
    boundary-condition id on a valid fluid cell, so treating nonzero as solid
    rejects inlet, outlet, moving walls, ground and the body surface -- which
    emptied the Q field completely.
    """
    # Loaded by path: importing xlb.cell_type would execute xlb/__init__.py and
    # drag in jax, which this suite deliberately avoids.
    cell_type = _load("xlb_cell_type_under_test", pathlib.Path("xlb") / "cell_type.py")

    assert lv._BC_SOLID == int(cell_type.BC_SOLID) == 255
    # And it must not collide with a level code or the boundary bit encoding.
    assert lv._BC_SOLID > lv._MAX_ENCODABLE_LEVELS


class _RateStub:
    """Records which half of maybe_render fired, without any GPU work."""

    def __init__(self, lv, fps=5.0, view_fps=30.0, heartbeat=1e9, max_lag_steps=0):
        self.enabled = True
        self.min_interval = 1.0 / fps
        self.min_view_interval = 1.0 / view_fps
        # Heartbeat off by default here so tests isolate camera-driven draws.
        self.heartbeat = heartbeat
        self.max_lag_steps = max_lag_steps
        self._last_sync = -(10**9)
        self.min_step_interval = 1
        self._last_data = 0.0
        self._last_render = 0.0
        self._last_step = -(10**9)
        self.state = lv._ViewerState(lv.Camera(target=[0.0, 0.0, 0.0], distance=1.0))
        self.camera_moved = False
        self.refreshes = 0
        self.draws = 0
        self.events = []

    # stand-ins for the real methods
    def _poll_camera(self):
        return self.camera_moved

    def _refresh_data(self, sim):
        self.refreshes += 1
        self.events.append("refresh")

    def _draw(self, step, mlups, hud):
        self.draws += 1
        self.events.append("draw")
        self._last_render = __import__("time").perf_counter()


def test_maybe_render_splits_data_refresh_from_camera_draw(lv):
    """The interactivity fix: a camera move must redraw without touching the solver.

    A data refresh runs sim.macro plus Neon staging and has to drain the solver's
    async queue, which is what made updates ~1/second. A camera-only draw reuses
    the resident Q field, so it must fire on its own faster clock and must NOT
    trigger a refresh.
    """
    s = _RateStub(lv, fps=5.0, view_fps=30.0)

    # First call: data is stale (never fetched), so both halves run.
    assert lv.LiveView.maybe_render(s, sim=None, step=0) is True
    assert (s.refreshes, s.draws) == (1, 1)

    # Immediately after, with no camera movement, nothing should happen.
    assert lv.LiveView.maybe_render(s, sim=None, step=1) is False
    assert (s.refreshes, s.draws) == (1, 1)

    # A camera move redraws only -- crucially without a refresh.
    s.camera_moved = True
    s._last_render = 0.0  # pretend the view interval has elapsed
    assert lv.LiveView.maybe_render(s, sim=None, step=2) is True
    assert s.refreshes == 1, "a camera move must not force a data refresh"
    assert s.draws == 2

    # Repeated moves inside the view interval are throttled.
    assert lv.LiveView.maybe_render(s, sim=None, step=3) is False
    assert s.draws == 2

    # Once the data interval elapses, a refresh happens again.
    s._last_data = 0.0
    s.camera_moved = False
    assert lv.LiveView.maybe_render(s, sim=None, step=4) is True
    assert (s.refreshes, s.draws) == (2, 3)


def test_maybe_render_draw_rate_can_exceed_data_rate(lv):
    """view_fps > fps is the point: many draws per data refresh.

    view_fps=0 means unthrottled, which is what lets this run without sleeping.
    """
    s = _RateStub(lv, fps=1.0, view_fps=1000.0)
    s.min_view_interval = 0.0  # unthrottled, as view_fps=0 would give

    lv.LiveView.maybe_render(s, sim=None, step=0)
    assert (s.refreshes, s.draws) == (1, 1)

    s.camera_moved = True
    for step in range(1, 21):
        lv.LiveView.maybe_render(s, sim=None, step=step)

    assert s.refreshes == 1, "no extra data refresh should have been triggered"
    assert s.draws == 21, f"every camera move should have redrawn, got {s.draws}"


def test_maybe_render_throttles_camera_draws_to_view_fps(lv):
    """Unbounded redraws would eat the GPU; view_fps has to actually bound them."""
    s = _RateStub(lv, fps=1.0, view_fps=30.0)

    lv.LiveView.maybe_render(s, sim=None, step=0)
    s.camera_moved = True

    # Many rapid calls inside one 1/30 s window must collapse to nothing extra.
    draws_before = s.draws
    for step in range(1, 101):
        lv.LiveView.maybe_render(s, sim=None, step=step)

    assert s.draws == draws_before, "draws inside the view interval must be suppressed"


def test_maybe_render_stops_when_viewer_closes(lv):
    s = _RateStub(lv)
    s.state.closed = True
    s._shutdown = lambda: setattr(s, "enabled", False)

    assert lv.LiveView.maybe_render(s, sim=None, step=0) is False
    assert s.enabled is False
    assert (s.refreshes, s.draws) == (0, 0)


def test_camera_draw_is_not_starved_by_a_perpetually_due_refresh(lv):
    """The starvation bug: a refresh costing more than its period is always due.

    With fps=5 the refresh is requested every 200 ms but actually takes seconds,
    so ``data_stale`` never goes false. Checking the refresh first then meant the
    cheap 3 ms camera path never ran and the window felt like it updated once a
    second. The draw has to be evaluated first.
    """
    s = _RateStub(lv, fps=5.0, view_fps=30.0)
    s.min_interval = 0.0  # refresh permanently due, as in the real failure

    s.camera_moved = True
    s._last_render = 0.0
    lv.LiveView.maybe_render(s, sim=None, step=0)

    # The decisive assertion is the *order*: the camera draw must be served
    # before the refresh, not after it (or not at all). A refresh always ends
    # with its own draw, so the count alone cannot distinguish the two orderings.
    assert s.events == ["draw", "refresh", "draw"], f"camera draw was starved: {s.events}"

    # Next call, inside the view interval: refresh still due, camera draw
    # throttled -- so only the refresh and its own draw.
    s.camera_moved = True
    s.events.clear()
    lv.LiveView.maybe_render(s, sim=None, step=1)
    assert s.events == ["refresh", "draw"], f"unexpected sequence: {s.events}"


def test_heartbeat_redraws_without_camera_movement(lv):
    """Without a heartbeat the published frame (and its age) goes stale when idle."""
    s = _RateStub(lv, fps=1000.0, view_fps=1000.0, heartbeat=0.0)
    s.min_interval = 1e9  # no refreshes at all
    s.min_view_interval = 0.0

    s.camera_moved = False
    for step in range(3):
        lv.LiveView.maybe_render(s, sim=None, step=step)

    assert s.refreshes == 0, "heartbeat must not trigger a data refresh"
    assert s.draws == 3, f"heartbeat should redraw when idle, got {s.draws}"


def test_heartbeat_disabled_leaves_idle_frames_alone(lv):
    s = _RateStub(lv, heartbeat=1e9)
    s.min_interval = 1e9
    s.min_view_interval = 0.0
    s.camera_moved = False

    for step in range(5):
        lv.LiveView.maybe_render(s, sim=None, step=step)

    assert (s.refreshes, s.draws) == (0, 0)


def test_max_lag_steps_bounds_host_run_ahead(lv, monkeypatch):
    """The main lever on refresh latency: keep CUDA's launch queue shallow.

    Without this the host fills the queue with hundreds of steps and a refresh
    has to wait out seconds of queued solver work.
    """
    syncs = []
    monkeypatch.setattr(lv.wp, "synchronize", lambda: syncs.append(1))

    s = _RateStub(lv, max_lag_steps=8)
    s.min_interval = 1e9  # isolate the sync from any refresh
    s.min_view_interval = 1e9

    for step in range(33):
        lv.LiveView.maybe_render(s, sim=None, step=step)

    # step 0 (first, since _last_sync starts at -1e9), then 8, 16, 24, 32.
    assert len(syncs) == 5, f"expected a sync every 8 steps, got {len(syncs)}"

    # Disabled means the solver's dispatch is left exactly as it was.
    syncs.clear()
    s2 = _RateStub(lv, max_lag_steps=0)
    s2.min_interval = 1e9
    s2.min_view_interval = 1e9
    for step in range(50):
        lv.LiveView.maybe_render(s2, sim=None, step=step)
    assert syncs == [], "max_lag_steps=0 must not sync at all"
