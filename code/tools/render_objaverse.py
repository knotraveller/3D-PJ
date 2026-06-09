#!/usr/bin/env python3
"""Batch render local Objaverse GLB files with Zero123++-style views.

Run with Blender, for example:
    blender -b --python render_objaverse.py -- --input_dir ./glbs --output_dir ./renders
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

UTILS_DIR = Path(__file__).resolve().parents[1] / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from camera_utils import save_cameras_json

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError as exc:  # pragma: no cover - only happens outside Blender.
    raise SystemExit(
        "render_objaverse.py must be run with Blender, e.g. "
        "`blender -b --python render_objaverse.py -- --input_dir ./glbs --output_dir ./renders`."
    ) from exc


RELATIVE_AZIMUTHS = [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]
TARGET_ELEVATIONS = [20.0, -10.0, 20.0, -10.0, 20.0, -10.0]
MIN_FOREGROUND_RATIO = 0.02
MAX_FOREGROUND_RATIO = 0.95


def clear_scene() -> None:
    """Remove all scene objects and purge unused data blocks."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # Keep memory bounded during large batches. Some Blender versions expose
    # different argument sets, so fall back to the no-argument operator.
    for _ in range(3):
        try:
            bpy.ops.outliner.orphans_purge(
                do_local_ids=True,
                do_linked_ids=True,
                do_recursive=True,
            )
        except TypeError:
            try:
                bpy.ops.outliner.orphans_purge()
            except Exception:
                break
        except Exception:
            break


def import_glb(path: Path) -> List[bpy.types.Object]:
    """Import a GLB file, remove imported cameras/lights, and return meshes."""
    bpy.ops.import_scene.gltf(filepath=str(path))

    for obj in list(bpy.context.scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.context.view_layer.update()
    mesh_objects = get_mesh_objects()
    if not mesh_objects:
        raise ValueError("Imported file contains no mesh objects.")
    return mesh_objects


def get_mesh_objects() -> List[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def vector_to_list(vec: Vector) -> List[float]:
    return [float(vec.x), float(vec.y), float(vec.z)]


def matrix_to_list(mat: Matrix) -> List[List[float]]:
    return [[float(mat[row][col]) for col in range(4)] for row in range(4)]


def compute_scene_bbox() -> Tuple[Vector, Vector]:
    """Compute the world-space bounding box over all mesh objects."""
    bpy.context.view_layer.update()
    mesh_objects = get_mesh_objects()
    if not mesh_objects:
        raise ValueError("No mesh objects found for bounding-box computation.")

    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))

    for obj in mesh_objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            bbox_min.x = min(bbox_min.x, world_corner.x)
            bbox_min.y = min(bbox_min.y, world_corner.y)
            bbox_min.z = min(bbox_min.z, world_corner.z)
            bbox_max.x = max(bbox_max.x, world_corner.x)
            bbox_max.y = max(bbox_max.y, world_corner.y)
            bbox_max.z = max(bbox_max.z, world_corner.z)

    return bbox_min, bbox_max


def normalize_scene(target_radius: float) -> Dict[str, object]:
    """Center meshes at the origin and scale the longest bbox side."""
    bbox_min, bbox_max = compute_scene_bbox()
    center = (bbox_min + bbox_max) * 0.5
    extent = bbox_max - bbox_min
    max_dim = max(float(extent.x), float(extent.y), float(extent.z))
    if max_dim <= 0.0 or not math.isfinite(max_dim):
        raise ValueError(f"Invalid scene bounding box with max_dim={max_dim}.")

    scale = (2.0 * float(target_radius)) / max_dim
    transform = Matrix.Scale(scale, 4) @ Matrix.Translation(-center)

    for obj in get_mesh_objects():
        obj.matrix_world = transform @ obj.matrix_world

    bpy.context.view_layer.update()
    normalized_min, normalized_max = compute_scene_bbox()

    return {
        "original_bbox": {
            "min": vector_to_list(bbox_min),
            "max": vector_to_list(bbox_max),
        },
        "normalized_bbox": {
            "min": vector_to_list(normalized_min),
            "max": vector_to_list(normalized_max),
        },
        "center": vector_to_list(center),
        "max_dim": max_dim,
        "scale": scale,
        "target_radius": float(target_radius),
    }


def set_eevee_engine(scene: bpy.types.Scene) -> None:
    for engine_name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = engine_name
            return
        except TypeError:
            continue
    raise ValueError("This Blender build does not expose an EEVEE render engine.")


def setup_renderer(resolution: int, engine: str) -> None:
    scene = bpy.context.scene
    engine = engine.upper()
    if engine == "CYCLES":
        scene.render.engine = "CYCLES"
        if hasattr(scene, "cycles"):
            scene.cycles.samples = 64
            scene.cycles.use_denoising = True
    else:
        set_eevee_engine(scene)
        if hasattr(scene, "eevee"):
            if hasattr(scene.eevee, "taa_render_samples"):
                scene.eevee.taa_render_samples = 64
            if hasattr(scene.eevee, "use_gtao"):
                scene.eevee.use_gtao = True

    scene.render.resolution_x = int(resolution)
    scene.render.resolution_y = int(resolution)
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"

    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (1.0, 1.0, 1.0)

    for owner, attr, value in (
        (scene.display_settings, "display_device", "sRGB"),
        (scene.view_settings, "view_transform", "Standard"),
        (scene.view_settings, "look", "None"),
        (scene.view_settings, "exposure", 0.0),
        (scene.view_settings, "gamma", 1.0),
    ):
        try:
            setattr(owner, attr, value)
        except Exception:
            pass


def setup_lighting() -> None:
    """Add fixed lights after imported lights have been removed."""
    for obj in list(bpy.context.scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.light_add(type="AREA", location=(0.0, -4.0, 5.0))
    area = bpy.context.object
    area.name = "Fixed_Area_Key"
    area.data.energy = 500.0
    area.data.size = 5.0
    look_at(area, (0.0, 0.0, 0.0))

    bpy.ops.object.light_add(type="SUN", location=(3.0, -4.0, 5.0))
    sun = bpy.context.object
    sun.name = "Fixed_Sun_Fill"
    sun.data.energy = 1.0
    look_at(sun, (0.0, 0.0, 0.0))


def create_camera(fov: float) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=(0.0, -4.0, 0.0))
    camera = bpy.context.object
    camera.name = "Render_Camera"
    camera.data.angle = math.radians(float(fov))
    camera.data.clip_start = 0.01
    camera.data.clip_end = 1000.0
    bpy.context.scene.camera = camera
    return camera


def set_camera_pose(
    camera: bpy.types.Object,
    azimuth: float,
    elevation: float,
    radius: float,
) -> None:
    """Place the camera with z-up spherical coordinates around the origin."""
    azimuth_rad = math.radians(float(azimuth))
    elevation_rad = math.radians(float(elevation))
    xy_radius = float(radius) * math.cos(elevation_rad)

    # Match camera_utils.py: azimuth=0 starts on world -Y.
    location = Vector(
        (
            xy_radius * math.sin(azimuth_rad),
            -xy_radius * math.cos(azimuth_rad),
            float(radius) * math.sin(elevation_rad),
        )
    )

    forward = (Vector((0.0, 0.0, 0.0)) - location).normalized()
    world_up = Vector((0.0, 0.0, 1.0))
    right = forward.cross(world_up)
    if right.length < 1e-8:
        right = forward.cross(Vector((0.0, 1.0, 0.0)))
    right.normalize()
    true_up = right.cross(forward).normalized()

    c2w = Matrix.Identity(4)
    c2w[0][0], c2w[1][0], c2w[2][0] = right.x, right.y, right.z
    c2w[0][1], c2w[1][1], c2w[2][1] = true_up.x, true_up.y, true_up.z
    c2w[0][2], c2w[1][2], c2w[2][2] = -forward.x, -forward.y, -forward.z
    c2w.translation = location
    camera.matrix_world = c2w
    bpy.context.view_layer.update()


def look_at(camera: bpy.types.Object, target: Sequence[float]) -> None:
    direction = Vector(target) - camera.location
    if direction.length == 0.0:
        raise ValueError("Cannot orient an object toward its own location.")
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def get_camera_matrices(camera: bpy.types.Object) -> Dict[str, List[List[float]]]:
    scene = bpy.context.scene
    bpy.context.view_layer.update()

    c2w = camera.matrix_world.copy()
    w2c = c2w.inverted()
    width = scene.render.resolution_x * scene.render.resolution_percentage / 100.0
    height = scene.render.resolution_y * scene.render.resolution_percentage / 100.0
    fx = 0.5 * width / math.tan(camera.data.angle_x * 0.5)
    fy = 0.5 * height / math.tan(camera.data.angle_y * 0.5)
    cx = width * 0.5
    cy = height * 0.5
    k = [
        [float(fx), 0.0, float(cx)],
        [0.0, float(fy), float(cy)],
        [0.0, 0.0, 1.0],
    ]

    return {
        "c2w": matrix_to_list(c2w),
        "w2c": matrix_to_list(w2c),
        "K": k,
    }


def render_rgba(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    return output_path


def save_png_image(
    image: bpy.types.Image,
    output_path: Path,
    color_mode: str,
) -> None:
    scene = bpy.context.scene
    settings = scene.render.image_settings
    old_file_format = settings.file_format
    old_color_mode = settings.color_mode
    old_color_depth = settings.color_depth

    output_path.parent.mkdir(parents=True, exist_ok=True)
    settings.file_format = "PNG"
    settings.color_mode = color_mode
    settings.color_depth = "8"
    image.save_render(str(output_path), scene=scene)

    settings.file_format = old_file_format
    settings.color_mode = old_color_mode
    settings.color_depth = old_color_depth


def save_rgb_and_alpha(rgba_path: Path, rgb_path: Path, alpha_path: Path) -> float:
    """Save white-background RGB and binary foreground mask from rendered RGBA."""
    rgba_image = bpy.data.images.load(str(rgba_path), check_existing=False)
    width, height = int(rgba_image.size[0]), int(rgba_image.size[1])
    pixels = list(rgba_image.pixels[:])

    rgb_pixels = [0.0] * len(pixels)
    alpha_pixels = [0.0] * len(pixels)
    foreground_sum = 0.0
    pixel_count = width * height

    for idx in range(0, len(pixels), 4):
        r = max(0.0, min(1.0, float(pixels[idx])))
        g = max(0.0, min(1.0, float(pixels[idx + 1])))
        b = max(0.0, min(1.0, float(pixels[idx + 2])))
        a = max(0.0, min(1.0, float(pixels[idx + 3])))

        rgb_pixels[idx] = a * r + (1.0 - a)
        rgb_pixels[idx + 1] = a * g + (1.0 - a)
        rgb_pixels[idx + 2] = a * b + (1.0 - a)
        rgb_pixels[idx + 3] = 1.0

        mask = 1.0 if a >= 0.5 else 0.0
        alpha_pixels[idx] = mask
        alpha_pixels[idx + 1] = mask
        alpha_pixels[idx + 2] = mask
        alpha_pixels[idx + 3] = mask
        foreground_sum += mask

    rgb_image = bpy.data.images.new("rgb_white", width=width, height=height, alpha=True)
    alpha_image = bpy.data.images.new("alpha_mask", width=width, height=height, alpha=True)
    rgb_image.pixels.foreach_set(rgb_pixels)
    alpha_image.pixels.foreach_set(alpha_pixels)
    rgb_image.update()
    alpha_image.update()

    save_png_image(rgb_image, rgb_path, color_mode="RGB")
    save_png_image(alpha_image, alpha_path, color_mode="RGBA")

    bpy.data.images.remove(rgba_image)
    bpy.data.images.remove(rgb_image)
    bpy.data.images.remove(alpha_image)

    try:
        rgba_path.unlink()
    except FileNotFoundError:
        pass

    return foreground_sum / float(pixel_count)


def view_camera_record(
    camera: bpy.types.Object,
    azimuth: float,
    elevation: float,
    radius: float,
    fov: float,
    index: Optional[int] = None,
    relative_azimuth: Optional[float] = None,
) -> Dict[str, object]:
    record: Dict[str, object] = {}
    if index is not None:
        record["index"] = int(index)
    if relative_azimuth is not None:
        record["relative_azimuth"] = float(relative_azimuth)
    record.update(
        {
            "azimuth": float(azimuth),
            "elevation": float(elevation),
            "radius": float(radius),
            "fov": float(fov),
        }
    )
    record.update(get_camera_matrices(camera))
    return record


def render_view(
    camera: bpy.types.Object,
    azimuth: float,
    elevation: float,
    radius: float,
    fov: float,
    rgba_path: Path,
    rgb_path: Path,
    alpha_path: Path,
    index: Optional[int] = None,
    relative_azimuth: Optional[float] = None,
) -> Tuple[Dict[str, object], float]:
    set_camera_pose(camera, azimuth, elevation, radius)
    render_rgba(rgba_path)
    foreground_ratio = save_rgb_and_alpha(rgba_path, rgb_path, alpha_path)
    record = view_camera_record(
        camera,
        azimuth=azimuth,
        elevation=elevation,
        radius=radius,
        fov=fov,
        index=index,
        relative_azimuth=relative_azimuth,
    )
    return record, foreground_ratio


def foreground_warnings(ratios: Iterable[float]) -> List[str]:
    values = list(ratios)
    warnings: List[str] = []
    if values and all(value < MIN_FOREGROUND_RATIO for value in values):
        warnings.append(
            f"all_foreground_ratios_below_{MIN_FOREGROUND_RATIO:g}"
        )
    if any(value > MAX_FOREGROUND_RATIO for value in values):
        warnings.append(
            f"foreground_ratio_above_{MAX_FOREGROUND_RATIO:g}"
        )
    return warnings


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sanitize_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._")
    return value or "asset"


def asset_output_name(glb_path: Path, input_dir: Path) -> str:
    try:
        relative = glb_path.relative_to(input_dir).with_suffix("")
        parts = [sanitize_component(part) for part in relative.parts]
        return "__".join(part for part in parts if part)
    except ValueError:
        return sanitize_component(glb_path.stem)


def format_ref_dir(azimuth: float) -> str:
    normalized = float(azimuth) % 360.0
    rounded = round(normalized)
    if abs(normalized - rounded) < 1e-6:
        return f"ref_{int(rounded):03d}"
    text = f"{normalized:07.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"ref_{text}"


def is_sample_complete(sample_dir: Path) -> bool:
    required = [
        sample_dir / "meta.json",
        sample_dir / "cameras.json",
        sample_dir / "cond" / "rgb.png",
        sample_dir / "cond" / "alpha.png",
    ]
    for idx in range(6):
        required.append(sample_dir / "targets" / f"{idx:03d}_rgb.png")
        required.append(sample_dir / "targets" / f"{idx:03d}_alpha.png")

    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def render_reference_sample(
    glb_path: Path,
    sample_dir: Path,
    theta0: float,
    normalization_meta: Dict[str, object],
    camera: bpy.types.Object,
    args: argparse.Namespace,
) -> Dict[str, object]:
    cond_dir = sample_dir / "cond"
    targets_dir = sample_dir / "targets"
    cond_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    foreground_ratios: Dict[str, object] = {}
    camera_specs: Dict[str, object] = {
        "resolution": int(args.resolution),
        "fov": float(args.fov),
        "radius": float(args.camera_radius),
        "views": [],
    }

    input_record, input_ratio = render_view(
        camera=camera,
        azimuth=theta0,
        elevation=args.input_elevation,
        radius=args.camera_radius,
        fov=args.fov,
        rgba_path=cond_dir / "_rgba.png",
        rgb_path=cond_dir / "rgb.png",
        alpha_path=cond_dir / "alpha.png",
        index=0,
        relative_azimuth=0.0,
    )
    camera_specs["views"].append({"name": "cond", **input_record})
    foreground_ratios["input"] = input_ratio

    target_ratios: List[float] = []
    for idx, (relative_azimuth, elevation) in enumerate(
        zip(RELATIVE_AZIMUTHS, TARGET_ELEVATIONS)
    ):
        target_azimuth = float(theta0) + relative_azimuth
        record, ratio = render_view(
            camera=camera,
            azimuth=target_azimuth,
            elevation=elevation,
            radius=args.camera_radius,
            fov=args.fov,
            rgba_path=targets_dir / f"{idx:03d}_rgba.png",
            rgb_path=targets_dir / f"{idx:03d}_rgb.png",
            alpha_path=targets_dir / f"{idx:03d}_alpha.png",
            index=idx + 1,
            relative_azimuth=relative_azimuth,
        )
        camera_specs["views"].append({"name": f"target_{idx:03d}", **record})
        target_ratios.append(ratio)

    foreground_ratios["targets"] = target_ratios
    all_ratios = [input_ratio] + target_ratios
    warnings = foreground_warnings(all_ratios)

    meta: Dict[str, object] = {
        "source_glb": str(glb_path),
        "reference_azimuth": float(theta0),
        "input_elevation": float(args.input_elevation),
        "relative_azimuths": RELATIVE_AZIMUTHS,
        "target_elevations": TARGET_ELEVATIONS,
        "resolution": int(args.resolution),
        "engine": str(args.engine).upper(),
        "normalization": normalization_meta,
        "foreground_ratios": foreground_ratios,
    }
    if warnings:
        meta["warning"] = warnings

    save_cameras_json(camera_specs, str(sample_dir / "cameras.json"))
    write_json(sample_dir / "meta.json", meta)

    return {
        "sample_dir": str(sample_dir),
        "reference_azimuth": float(theta0),
        "foreground_ratios": foreground_ratios,
        "warning": warnings,
    }


def process_one_glb(glb_path: Path, output_dir: Path, args: argparse.Namespace) -> Dict[str, object]:
    input_dir = Path(args.input_dir)
    asset_dir = output_dir / asset_output_name(glb_path, input_dir)
    sample_dirs = [asset_dir / format_ref_dir(theta0) for theta0 in args.ref_azimuths]

    if args.skip_existing and all(is_sample_complete(sample_dir) for sample_dir in sample_dirs):
        return {
            "file": str(glb_path),
            "asset_dir": str(asset_dir),
            "status": "skipped_existing",
            "samples": [str(sample_dir) for sample_dir in sample_dirs],
        }

    clear_scene()
    import_glb(glb_path)
    normalization_meta = normalize_scene(args.target_radius)
    setup_renderer(args.resolution, args.engine)
    setup_lighting()
    camera = create_camera(args.fov)

    rendered_samples = []
    skipped_samples = []
    for theta0, sample_dir in zip(args.ref_azimuths, sample_dirs):
        if args.skip_existing and is_sample_complete(sample_dir):
            skipped_samples.append(str(sample_dir))
            continue
        rendered_samples.append(
            render_reference_sample(
                glb_path=glb_path,
                sample_dir=sample_dir,
                theta0=theta0,
                normalization_meta=normalization_meta,
                camera=camera,
                args=args,
            )
        )

    return {
        "file": str(glb_path),
        "asset_dir": str(asset_dir),
        "status": "rendered",
        "rendered_samples": rendered_samples,
        "skipped_samples": skipped_samples,
    }


def discover_glbs(input_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".glb"
        ],
        key=lambda path: str(path).lower(),
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    if argv is None:
        if "--" in sys.argv:
            argv = sys.argv[sys.argv.index("--") + 1 :]
        else:
            argv = []

    parser = argparse.ArgumentParser(
        description="Render local GLB files into Zero123++-style multi-view data."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing .glb files.")
    parser.add_argument("--output_dir", required=True, help="Output render directory.")
    parser.add_argument("--resolution", type=int, default=256, help="Square render resolution.")
    parser.add_argument(
        "--ref_azimuths",
        type=float,
        nargs="+",
        default=[0.0, 90.0, 180.0, 270.0],
        help="Input/reference azimuth list in degrees.",
    )
    parser.add_argument(
        "--input_elevation",
        type=float,
        default=0.0,
        help="Input/reference elevation in degrees.",
    )
    parser.add_argument("--fov", type=float, default=30.0, help="Camera field of view in degrees.")
    parser.add_argument(
        "--camera_radius",
        type=float,
        default=4.0,
        help="Camera radius around the normalized object.",
    )
    parser.add_argument(
        "--target_radius",
        type=float,
        default=0.8,
        help="Normalize longest object side to 2 * target_radius.",
    )
    parser.add_argument(
        "--engine",
        type=lambda value: value.upper(),
        choices=["EEVEE", "CYCLES"],
        default="EEVEE",
        help="Render engine.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a GLB if every requested ref view is already complete.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Only process the first N GLB files.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if args.resolution <= 0:
        raise SystemExit("--resolution must be positive.")
    if args.camera_radius <= 0.0:
        raise SystemExit("--camera_radius must be positive.")
    if args.target_radius <= 0.0:
        raise SystemExit("--target_radius must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_log = output_dir / "rendered.jsonl"
    failed_log = output_dir / "failed.jsonl"

    glb_files = discover_glbs(input_dir)
    if args.max_files is not None:
        glb_files = glb_files[: max(args.max_files, 0)]

    if not glb_files:
        print(f"No .glb files found under {input_dir}")
        return

    total = len(glb_files)
    for index, glb_path in enumerate(glb_files, start=1):
        print(f"[{index}/{total}] Rendering {glb_path}")
        try:
            record = process_one_glb(glb_path, output_dir, args)
        except Exception as exc:
            failure = {
                "file": str(glb_path),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(failed_log, failure)
            print(f"  FAILED: {exc}")
            continue

        append_jsonl(rendered_log, record)
        print(f"  {record['status']}: {record.get('asset_dir', '')}")

    clear_scene()
    print(f"Done. Logs: {rendered_log} / {failed_log}")


if __name__ == "__main__":
    main()
