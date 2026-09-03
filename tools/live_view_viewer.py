#!/usr/bin/env python3
"""Interactive viewer for an XLB live_view frame directory.

Why this exists
---------------
The solver often runs somewhere that cannot open a window: a Docker container
with no X11 socket, a headless node, or WSL where CUDA/OpenGL interop is
unsupported. But the frame directory is usually visible from somewhere that
*can* -- a bind-mounted host, a network share.

So this viewer talks to the running solve entirely through that directory:

    status.json   written by LiveView   -> newest frame, HUD numbers
    live_NNNNNN.png                     -> the frames themselves
    camera.json   written by this tool  -> camera / iso commands

No network, no X11 forwarding, no OpenGL. It uses only the standard library
(tkinter, with Tk 8.6 loading the PNGs directly), so there is nothing to install
on the viewing machine.

Usage
-----
    python tools/live_view_viewer.py [FRAME_DIR]

With no argument it finds the most recently updated ``live_view`` directory
under the current tree. Typical use from Windows against a container-side run:

    python tools/live_view_viewer.py "C:\\Work\\XLB\\SWT\\XLB\\examples\\cfd\\stl-files\\R2 16mm LiveView\\live_view"

Controls
--------
    drag         orbit
    wheel        zoom
    [ / ]        lower / raise the Q iso level
    r            recalibrate the iso level from the current field
    s            save a copy of the current frame next to the viewer
    q / Escape   quit the viewer (the solve keeps running)

Camera changes take effect on the solver's next frame, so at the default 10 fps
expect up to ~100 ms of lag. That is the render cadence, not this tool.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import tkinter as tk
from pathlib import Path

# How often to look for a new frame. Faster than the solver's draw rate so the
# viewer never adds a frame of latency of its own.
_POLL_MS = 16

# Do not rewrite camera.json more often than this. Each write crosses a bind
# mount; the solver picks it up on its next draw.
_CAMERA_WRITE_INTERVAL = 0.02


def find_frame_dir(root="."):
    """Most recently updated live_view directory under ``root``."""
    candidates = []
    for path in Path(root).rglob("live_view"):
        if not path.is_dir():
            continue
        status = path / "status.json"
        stamp = status.stat().st_mtime if status.exists() else 0.0
        if stamp == 0.0:
            frames = list(path.glob("live_*.png"))
            if not frames:
                continue
            stamp = max(f.stat().st_mtime for f in frames)
        candidates.append((stamp, path))

    if not candidates:
        return None
    return max(candidates)[1]


class Viewer:
    def __init__(self, frame_dir):
        self.dir = Path(frame_dir)
        self.status_path = self.dir / "status.json"
        self.camera_path = self.dir / "camera.json"

        # Local camera state. Seeded from the solver's status on first contact so
        # dragging starts from whatever it is actually showing.
        self.azimuth = None
        self.elevation = None
        self.distance = None
        self.fov = None
        self.iso_scale = 1.0
        self.recalibrate_seq = 0

        self.last_frame = -1
        self.last_camera_write = 0.0
        self.camera_dirty = False
        self.current_frame_path = None
        self._image = None  # PhotoImage must be retained or Tk garbage-collects it

        self.root = tk.Tk()
        self.root.title(f"XLB live view - {self.dir.name}")
        self.root.configure(bg="black")

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0, width=1280, height=720)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.image_id = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.hud_id = self.canvas.create_text(10, 10, anchor=tk.NW, fill="#e0e0e0", font=("Consolas", 10), text="waiting for first frame...")

        self._bind()

    # -- input ---------------------------------------------------------------

    def _bind(self):
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.root.bind("<Key>", self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

    def _on_press(self, event):
        self._drag_from = (event.x, event.y)

    def _on_drag(self, event):
        if self.azimuth is None:
            return  # no status yet, nothing to orbit relative to

        x0, y0 = self._drag_from
        self.azimuth -= (event.x - x0) * 0.4
        self.elevation = max(-85.0, min(85.0, self.elevation + (event.y - y0) * 0.4))
        self._drag_from = (event.x, event.y)
        self.camera_dirty = True

    def _on_wheel(self, event):
        if self.distance is None:
            return
        # Windows reports delta in multiples of 120; other platforms use +-1.
        notches = event.delta / 120.0 if abs(event.delta) >= 120 else float(event.delta)
        self.distance *= 0.9**notches
        self.camera_dirty = True

    def _on_key(self, event):
        key = event.keysym.lower()
        if key in ("q", "escape"):
            self._on_quit()
        elif key == "bracketleft":
            self.iso_scale /= 1.3
            self.camera_dirty = True
        elif key == "bracketright":
            self.iso_scale *= 1.3
            self.camera_dirty = True
        elif key == "r":
            self.recalibrate_seq += 1
            self.camera_dirty = True
        elif key == "s":
            self._save_copy()

    def _on_quit(self):
        self.root.destroy()

    def _save_copy(self):
        if self.current_frame_path is None:
            return
        dest = Path.cwd() / f"live_view_snapshot_{self.last_frame:06d}.png"
        try:
            shutil.copyfile(self.current_frame_path, dest)
            print(f"saved {dest}")
        except OSError as exc:
            print(f"could not save snapshot: {exc}")

    # -- solver bridge -------------------------------------------------------

    def _write_camera(self):
        """Publish camera state, replacing atomically so the solver never reads a
        half-written file."""
        now = time.time()
        if not self.camera_dirty or now - self.last_camera_write < _CAMERA_WRITE_INTERVAL:
            return

        payload = {
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "distance": self.distance,
            "fov": self.fov,
            "iso_scale": self.iso_scale,
            "recalibrate_seq": self.recalibrate_seq,
        }

        tmp = self.camera_path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, self.camera_path)
        except OSError as exc:
            print(f"could not write camera.json: {exc}")
            return

        self.last_camera_write = now
        self.camera_dirty = False

    def _read_status(self):
        try:
            return json.loads(self.status_path.read_text())
        except (OSError, ValueError):
            # Missing or mid-replace; try again on the next tick.
            return None

    def _adopt_camera(self, status):
        """Take the solver's camera as our starting point, once."""
        if self.azimuth is not None:
            return
        self.azimuth = float(status.get("azimuth", 35.0))
        self.elevation = float(status.get("elevation", 20.0))
        self.distance = float(status.get("distance", 1.0))
        self.fov = float(status.get("fov", 45.0))
        self.iso_scale = float(status.get("iso_scale", 1.0))

    # -- main loop -----------------------------------------------------------

    def _tick(self):
        status = self._read_status()

        if status is not None:
            self._adopt_camera(status)

            frame = int(status.get("frame", -1))
            if frame >= 0 and frame != self.last_frame:
                path = self.dir / status.get("frame_file", f"live_{frame:06d}.png")
                if path.exists():
                    try:
                        image = tk.PhotoImage(file=str(path))
                    except tk.TclError:
                        # Frame still being written; leave the previous one up.
                        image = None

                    if image is not None:
                        self._image = image
                        self.canvas.itemconfigure(self.image_id, image=image)
                        self.canvas.configure(width=image.width(), height=image.height())
                        self.last_frame = frame
                        self.current_frame_path = path

            age = time.time() - float(status.get("time", 0.0))
            self.canvas.itemconfigure(
                self.hud_id,
                text=(
                    f"frame {self.last_frame}   step {status.get('step', '?')}   "
                    f"MLUPS {float(status.get('mlups', 0.0)):.0f}   "
                    f"render {float(status.get('render_ms', 0.0)):.0f} ms   age {age:.1f}s\n"
                    f"q_iso {float(status.get('q_iso', 0.0)):.3g}   iso_scale {self.iso_scale:.2f}   "
                    f"grid {'x'.join(str(v) for v in status.get('render_shape', []))}\n"
                    f"drag orbit | wheel zoom | [ ] iso | r recalibrate | s save | q quit"
                ),
            )
        elif not self.dir.is_dir():
            self.canvas.itemconfigure(
                self.hud_id,
                text=f"waiting for the solve to create\n{self.dir}\n\n(mesh prep runs before the first frame)",
            )
        else:
            self.canvas.itemconfigure(
                self.hud_id,
                text=f"waiting for first frame in\n{self.dir}\n\nno status.json yet",
            )

        self._write_camera()
        self.root.after(_POLL_MS, self._tick)

    def run(self):
        self.root.after(_POLL_MS, self._tick)
        self.root.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "frame_dir",
        nargs="?",
        help="live_view directory to watch; searched for under the current tree if omitted",
    )
    args = parser.parse_args(argv)

    frame_dir = args.frame_dir or find_frame_dir()
    if frame_dir is None:
        parser.error("no live_view directory found; pass one explicitly")

    frame_dir = Path(frame_dir)
    if frame_dir.exists() and not frame_dir.is_dir():
        parser.error(f"not a directory: {frame_dir}")

    # Deliberately tolerate a directory that does not exist yet. A solve spends
    # a long time in mesh prep before the first frame, and being able to start
    # the viewer while it spools up is the normal case, not an edge case.
    if not frame_dir.exists():
        print(f"waiting for {frame_dir} (not created yet)")
    else:
        print(f"watching {frame_dir}")
    Viewer(frame_dir).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
