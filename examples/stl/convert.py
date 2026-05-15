#!/usr/bin/env python3

"""
Streaming OBJ -> Binary STL converter.

Designed for very large OBJ files.

Key properties:
  - No trimesh
  - No Open3D
  - Does not store faces
  - Stores only vertices as float32
  - Streams triangles directly to binary STL
  - Applies scale during vertex ingestion
  - Uses large buffered writes
  - Optionally computes real STL normals

Example:
  python obj_to_binary_stl_stream.py RivianR2_63M.obj body_scaled.stl --scale 0.01

With real normals:
  python obj_to_binary_stl_stream.py RivianR2_63M.obj body_scaled.stl --scale 0.01 --normals

With a pre-count pass:
  python obj_to_binary_stl_stream.py RivianR2_63M.obj body_scaled.stl --scale 0.01 --precount
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from array import array
from pathlib import Path


TRI_STRUCT = struct.Struct("<12fH")
# Binary STL triangle layout:
#   normal: 3 float32
#   v0:     3 float32
#   v1:     3 float32
#   v2:     3 float32
#   attr:   uint16
#
# Total = 12 floats * 4 bytes + 2 bytes = 50 bytes.


def is_vertex_line(line: bytes) -> bool:
    return len(line) >= 2 and line[0] == 118 and line[1] in (32, 9)
    # 118 = b"v", 32 = space, 9 = tab


def is_face_line(line: bytes) -> bool:
    return len(line) >= 2 and line[0] == 102 and line[1] in (32, 9)
    # 102 = b"f"


def strip_leading_ws_if_needed(line: bytes) -> bytes:
    """
    Most OBJ files have no leading whitespace, so avoid lstrip() on the fast path.
    """
    if line and line[0] in (32, 9, 13):
        return line.lstrip()
    return line


def parse_obj_index(token: bytes, vertex_count: int) -> int:
    """
    Parse an OBJ face vertex token.

    Supports:
      b"123"
      b"123/45"
      b"123//67"
      b"123/45/67"
      b"-1"
      b"-1/-2/-3"

    OBJ positive indices are 1-based.
    OBJ negative indices are relative to the current vertex count.
    """

    slash = token.find(b"/")
    if slash != -1:
        token = token[:slash]

    if not token:
        raise ValueError("empty OBJ face index")

    raw = int(token)

    if raw > 0:
        idx = raw - 1
    elif raw < 0:
        idx = vertex_count + raw
    else:
        raise ValueError("OBJ index 0 is invalid")

    if idx < 0 or idx >= vertex_count:
        raise IndexError(
            f"OBJ face index {raw} resolved to invalid zero-based index {idx}; "
            f"current vertex count is {vertex_count}"
        )

    return idx


def parse_vertex_line(line: bytes, scale: float) -> tuple[float, float, float]:
    """
    Parse a vertex line.

    Example:
      b"v 1.0 2.0 3.0"
      b"v 1.0 2.0 3.0 1.0"  # optional vertex color/weight ignored
    """

    parts = line.split(maxsplit=4)

    if len(parts) < 4:
        raise ValueError(f"invalid vertex line: {line[:120]!r}")

    x = float(parts[1]) * scale
    y = float(parts[2]) * scale
    z = float(parts[3]) * scale

    return x, y, z


def compute_normal(
    ax: float, ay: float, az: float,
    bx: float, by: float, bz: float,
    cx: float, cy: float, cz: float,
) -> tuple[float, float, float]:
    """
    Scalar normal calculation.

    This avoids per-triangle NumPy calls, which are expensive when repeated
    tens of millions of times.
    """

    ux = bx - ax
    uy = by - ay
    uz = bz - az

    vx = cx - ax
    vy = cy - ay
    vz = cz - az

    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx

    length_sq = nx * nx + ny * ny + nz * nz

    if length_sq > 0.0:
        inv_len = 1.0 / math.sqrt(length_sq)
        return nx * inv_len, ny * inv_len, nz * inv_len

    return 0.0, 0.0, 0.0


def count_obj_vertices_and_triangles(obj_path: Path) -> tuple[int, int]:
    """
    Optional pre-pass.

    Pros:
      - prints useful upfront counts
      - verifies triangle count before writing
      - allows early STL uint32 triangle-count check

    Cons:
      - reads the large OBJ one extra time
    """

    vertex_count = 0
    triangle_count = 0

    with obj_path.open("rb", buffering=16 * 1024 * 1024) as f:
        for raw_line in f:
            line = strip_leading_ws_if_needed(raw_line)

            if is_vertex_line(line):
                vertex_count += 1

            elif is_face_line(line):
                tokens = line.split()[1:]

                n = 0
                for tok in tokens:
                    if tok.startswith(b"#"):
                        break
                    n += 1

                if n >= 3:
                    triangle_count += n - 2

    return vertex_count, triangle_count


def convert_obj_to_binary_stl(
    obj_path: str | Path,
    stl_path: str | Path,
    scale: float = 0.01,
    compute_normals: bool = False,
    batch_triangles: int = 250_000,
    progress_every: int = 5_000_000,
    precount: bool = False,
) -> None:
    obj_path = Path(obj_path)
    stl_path = Path(stl_path)

    if batch_triangles <= 0:
        raise ValueError("batch_triangles must be positive")

    if progress_every < 0:
        raise ValueError("progress_every must be >= 0")

    if not obj_path.exists():
        raise FileNotFoundError(obj_path)

    expected_vertices = None
    expected_triangles = None

    if precount:
        print("Pre-counting vertices and triangles...", file=sys.stderr, flush=True)
        t0 = time.perf_counter()

        expected_vertices, expected_triangles = count_obj_vertices_and_triangles(obj_path)

        elapsed = time.perf_counter() - t0

        print(
            f"Pre-count complete in {elapsed:.1f}s: "
            f"vertices={expected_vertices:,}, triangles={expected_triangles:,}",
            file=sys.stderr,
            flush=True,
        )

        if expected_triangles > 0xFFFFFFFF:
            raise OverflowError(
                "Binary STL stores triangle count as uint32; "
                f"{expected_triangles:,} triangles is too many."
            )

    # Compact vertex storage.
    #
    # array("f") stores C float, usually IEEE float32.
    # Layout:
    #   [x0, y0, z0, x1, y1, z1, ...]
    #
    # RAM:
    #   vertex_count * 3 * 4 bytes
    vertices = array("f")

    # STL output buffer.
    #
    # Each triangle is 50 bytes.
    # 250,000 triangles = 12.5 MB buffer.
    tri_buffer = bytearray(batch_triangles * 50)
    buffer_offset = 0

    vertex_count = 0
    triangle_count = 0
    line_count = 0

    start_time = time.perf_counter()

    face_indices: list[int] = []

    def flush_buffer(out_file) -> None:
        nonlocal buffer_offset

        if buffer_offset:
            out_file.write(memoryview(tri_buffer)[:buffer_offset])
            buffer_offset = 0

    def get_vertex(idx: int) -> tuple[float, float, float]:
        base = idx * 3
        return vertices[base], vertices[base + 1], vertices[base + 2]

    def write_triangle(out_file, ia: int, ib: int, ic: int) -> None:
        nonlocal buffer_offset, triangle_count

        if buffer_offset + 50 > len(tri_buffer):
            flush_buffer(out_file)

        ax, ay, az = get_vertex(ia)
        bx, by, bz = get_vertex(ib)
        cx, cy, cz = get_vertex(ic)

        if compute_normals:
            nx, ny, nz = compute_normal(
                ax, ay, az,
                bx, by, bz,
                cx, cy, cz,
            )
        else:
            # Fast path. Most STL consumers recompute normals.
            nx = ny = nz = 0.0

        TRI_STRUCT.pack_into(
            tri_buffer,
            buffer_offset,

            nx, ny, nz,

            ax, ay, az,
            bx, by, bz,
            cx, cy, cz,

            0,
        )

        buffer_offset += 50
        triangle_count += 1

        if progress_every and triangle_count % progress_every == 0:
            elapsed = time.perf_counter() - start_time
            rate = triangle_count / elapsed if elapsed > 0 else 0.0

            print(
                f"triangles={triangle_count:,} "
                f"vertices={vertex_count:,} "
                f"lines={line_count:,} "
                f"rate={rate:,.0f} tri/s",
                file=sys.stderr,
                flush=True,
            )

    with obj_path.open("rb", buffering=16 * 1024 * 1024) as inp, \
         stl_path.open("wb", buffering=16 * 1024 * 1024) as out:

        header = (
            b"binary STL generated by streaming Python OBJ converter"
            b" scale="
            + str(scale).encode("ascii")
        )
        out.write(header[:80].ljust(80, b"\0"))

        # Placeholder triangle count.
        # We seek back and write the real count at the end.
        out.write(struct.pack("<I", 0))

        for raw_line in inp:
            line_count += 1

            line = strip_leading_ws_if_needed(raw_line)

            if not line or line.startswith(b"#"):
                continue

            if is_vertex_line(line):
                try:
                    x, y, z = parse_vertex_line(line, scale)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse vertex near line {line_count}: {exc}"
                    ) from exc

                vertices.append(x)
                vertices.append(y)
                vertices.append(z)
                vertex_count += 1

            elif is_face_line(line):
                tokens = line.split()[1:]

                face_indices.clear()

                try:
                    for tok in tokens:
                        if tok.startswith(b"#"):
                            break

                        idx = parse_obj_index(tok, vertex_count)
                        face_indices.append(idx)

                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse face near line {line_count}: {exc}"
                    ) from exc

                if len(face_indices) < 3:
                    continue

                root = face_indices[0]

                # Fan triangulation:
                #   f a b c       -> abc
                #   f a b c d     -> abc, acd
                #   f a b c d e   -> abc, acd, ade
                for i in range(1, len(face_indices) - 1):
                    write_triangle(
                        out,
                        root,
                        face_indices[i],
                        face_indices[i + 1],
                    )

                    if triangle_count > 0xFFFFFFFF:
                        raise OverflowError(
                            "Binary STL stores triangle count as uint32; "
                            "too many triangles."
                        )

        flush_buffer(out)

        out.seek(80)
        out.write(struct.pack("<I", triangle_count))

    elapsed = time.perf_counter() - start_time

    print("Done.", file=sys.stderr)
    print(f"Input:      {obj_path}", file=sys.stderr)
    print(f"Output:     {stl_path}", file=sys.stderr)
    print(f"Scale:      {scale}", file=sys.stderr)
    print(f"Normals:    {'computed' if compute_normals else 'zero'}", file=sys.stderr)
    print(f"Vertices:   {vertex_count:,}", file=sys.stderr)
    print(f"Triangles:  {triangle_count:,}", file=sys.stderr)
    print(f"Elapsed:    {elapsed:.1f}s", file=sys.stderr)

    if elapsed > 0:
        print(
            f"Rate:       {triangle_count / elapsed:,.0f} triangles/s",
            file=sys.stderr,
        )

    if expected_vertices is not None and expected_vertices != vertex_count:
        print(
            f"Warning: pre-counted {expected_vertices:,} vertices, "
            f"but converted {vertex_count:,}.",
            file=sys.stderr,
        )

    if expected_triangles is not None and expected_triangles != triangle_count:
        print(
            f"Warning: pre-counted {expected_triangles:,} triangles, "
            f"but wrote {triangle_count:,}.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a large OBJ directly to binary STL with optional scaling, "
            "without Trimesh or Open3D."
        )
    )

    parser.add_argument(
        "input_obj",
        help="Input OBJ file",
    )

    parser.add_argument(
        "output_stl",
        help="Output binary STL file",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="Scale factor applied to vertices during conversion. Default: 0.01",
    )

    parser.add_argument(
        "--normals",
        action="store_true",
        help=(
            "Compute and write real STL normals. Slower. "
            "Default is zero normals for speed."
        ),
    )

    parser.add_argument(
        "--batch-triangles",
        type=int,
        default=250_000,
        help=(
            "Number of triangles buffered before writing. "
            "Default: 250000, about 12.5 MB."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=5_000_000,
        help=(
            "Print progress every N triangles. "
            "Use 0 to disable. Default: 5000000."
        ),
    )

    parser.add_argument(
        "--precount",
        action="store_true",
        help=(
            "Do an initial pass to count vertices and triangles. "
            "Useful for validation, but reads the OBJ one extra time."
        ),
    )

    args = parser.parse_args()

    convert_obj_to_binary_stl(
        obj_path=args.input_obj,
        stl_path=args.output_stl,
        scale=args.scale,
        compute_normals=args.normals,
        batch_triangles=args.batch_triangles,
        progress_every=args.progress_every,
        precount=args.precount,
    )


if __name__ == "__main__":
    main()