from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import arcpy
import numpy as np
from osgeo import gdal


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_OUT = ROOT / "processed_hsi_uda"


@dataclass
class PatchInfo:
    file: str
    source: str
    x: int
    y: int
    size: int
    fg_ratio: float | None = None


def find_first(root: Path, pattern: str) -> Path:
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"No match for {pattern} under {root}")
    return hits[0]


def find_scene_dirs(root: Path, prefix: str) -> list[Path]:
    dirs = {p.parent for p in root.rglob(f"{prefix}*B01_CMOS3.tif")}
    return sorted(dirs)


def band_index(path: Path) -> int:
    match = re.search(r"_B(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot parse band index from {path}")
    return int(match.group(1))


def scene_band_files(scene_dir: Path) -> list[Path]:
    files = sorted(scene_dir.glob("*_B*_CMOS3.tif"), key=band_index)
    if len(files) != 32:
        raise RuntimeError(f"Expected 32 bands in {scene_dir}, got {len(files)}")
    return files


def ensure_dir(path: Path, clean: bool = False) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def rasterize_polygon_labels(
    polygon_path: Path,
    template_raster: arcpy.Raster,
    out_path: Path,
    field: str = "gridcode_1",
    name: str = "labels",
) -> Path:
    arcpy.env.overwriteOutput = True
    arcpy.env.outputCoordinateSystem = template_raster.spatialReference
    arcpy.env.snapRaster = template_raster
    arcpy.env.cellSize = template_raster.meanCellWidth
    in_path = polygon_path
    src_sr = arcpy.Describe(str(polygon_path)).spatialReference
    tpl_sr = template_raster.spatialReference
    src_code = getattr(src_sr, "factoryCode", None)
    tpl_code = getattr(tpl_sr, "factoryCode", None)
    if src_code not in (None, 0) and tpl_code not in (None, 0) and src_code != tpl_code:
        scratch_gdb = Path(arcpy.env.scratchGDB)
        proj_fc = scratch_gdb / f"{name}_proj"
        if arcpy.Exists(str(proj_fc)):
            arcpy.management.Delete(str(proj_fc))
        arcpy.management.Project(str(polygon_path), str(proj_fc), tpl_sr)
        in_path = proj_fc
    square_path = out_path.with_name(f"{out_path.stem}_square.tif")
    try:
        arcpy.conversion.PolygonToRaster(
            str(in_path),
            field,
            str(square_path),
            cell_assignment="MAXIMUM_AREA",
            priority_field=field,
            cellsize=template_raster.meanCellWidth,
        )
    except Exception:
        arcpy.conversion.PolygonToRaster(
            str(in_path),
            field,
            str(square_path),
            cell_assignment="CELL_CENTER",
            priority_field=field,
            cellsize=template_raster.meanCellWidth,
        )
    x_res = abs(float(template_raster.meanCellWidth))
    y_res = abs(float(template_raster.meanCellHeight))
    x_min = float(template_raster.extent.XMin)
    y_max = float(template_raster.extent.YMax)
    x_max = x_min + int(template_raster.width) * x_res
    y_min = y_max - int(template_raster.height) * y_res
    source = gdal.Open(str(square_path))
    source_nodata = source.GetRasterBand(1).GetNoDataValue() if source is not None else None
    source = None
    gdal.Warp(
        str(out_path),
        str(square_path),
        options=gdal.WarpOptions(
            format="GTiff",
            outputBounds=(x_min, y_min, x_max, y_max),
            xRes=x_res,
            yRes=y_res,
            resampleAlg="near",
            srcNodata=source_nodata if source_nodata is not None else 255,
            dstNodata=255,
            outputType=gdal.GDT_Byte,
            creationOptions=["COMPRESS=LZW"],
        ),
    )
    square_path.unlink(missing_ok=True)
    return out_path


def composite_bands(scene_dir: Path, out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    arcpy.management.CompositeBands([str(p) for p in scene_band_files(scene_dir)], str(out_path))
    return out_path


def read_window(
    raster: arcpy.Raster,
    x: int,
    y: int,
    size: int,
    reference_raster: arcpy.Raster | None = None,
    nodata_to_value=None,
) -> np.ndarray:
    ref = reference_raster if reference_raster is not None else raster
    cell_x = abs(float(ref.meanCellWidth))
    cell_y = abs(float(ref.meanCellHeight))
    x0 = ref.extent.XMin + x * cell_x
    y0 = ref.extent.YMax - (y + size) * cell_y
    kwargs = {}
    if nodata_to_value is not None:
        kwargs["nodata_to_value"] = nodata_to_value
    arr = arcpy.RasterToNumPyArray(raster, arcpy.Point(x0, y0), ncols=size, nrows=size, **kwargs)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return np.asarray(arr)


def fold_id(x: int, y: int, stride: int) -> int:
    return ((x // stride) // 2 + 2 * ((y // stride) // 2)) % 5


def save_patch(
    out_dir: Path,
    prefix: str,
    x: int,
    y: int,
    img: np.ndarray,
    mask: np.ndarray | None,
    source: str,
) -> PatchInfo:
    path = out_dir / f"{prefix}_x{x:05d}_y{y:05d}.npz"
    payload = {
        "image": np.asarray(img, dtype=np.uint16),
        "source": source,
        "x": np.int32(x),
        "y": np.int32(y),
    }
    fg_ratio = None
    if mask is not None:
        mask = np.asarray(mask, dtype=np.uint8)
        payload["mask"] = mask
        valid = mask != 255
        fg_ratio = float(np.count_nonzero((mask > 0) & valid) / max(int(valid.sum()), 1))
    np.savez_compressed(path, **payload)
    return PatchInfo(
        file=str(path.resolve()),
        source=source,
        x=x,
        y=y,
        size=img.shape[-1],
        fg_ratio=fg_ratio,
    )


def export_patches(
    image_raster: arcpy.Raster,
    label_raster: arcpy.Raster | None,
    out_dir: Path,
    prefix: str,
    split: str,
    patch_size: int,
    stride: int,
    source: str,
    min_fg_pixels: int = 64,
    min_image_fraction: float = 0.05,
) -> list[PatchInfo]:
    infos: list[PatchInfo] = []
    width = int(image_raster.width)
    height = int(image_raster.height)
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            fold = fold_id(x, y, stride)
            if split == "train" and fold == 0:
                continue
            if split == "val" and fold != 0:
                continue
            img = read_window(image_raster, x, y, patch_size)
            image_fraction = float(np.count_nonzero(img) / max(img.size, 1))
            if image_fraction < min_image_fraction:
                continue
            mask = None
            if label_raster is not None:
                mask = read_window(
                    label_raster,
                    x,
                    y,
                    patch_size,
                    reference_raster=image_raster,
                    nodata_to_value=255,
                )[0]
                mask = np.asarray(mask, dtype=np.int64)
                mask[~np.isin(mask, [0, 1, 2, 3])] = 255
                valid = mask != 255
                valid_pixels = int(np.count_nonzero(valid))
                fg_pixels = int(np.count_nonzero((mask > 0) & valid))
                if split == "train" and fg_pixels < min_fg_pixels:
                    continue
                if split == "all" and valid_pixels < min_fg_pixels:
                    continue
            infos.append(save_patch(out_dir, prefix, x, y, img, mask, source))
    return infos


def merge_shapefiles(shps: list[Path], out_path: Path) -> Path:
    if out_path.exists():
        return out_path
    arcpy.management.Merge([str(p) for p in shps], str(out_path))
    return out_path


def compute_stats(npz_paths: list[Path]) -> dict:
    sum_ = None
    sumsq = None
    count = 0
    class_counts: dict[int, int] = {}
    for path in npz_paths:
        data = np.load(path)
        img = data["image"].astype(np.float64)
        c, h, w = img.shape
        flat = img.reshape(c, -1)
        if sum_ is None:
            sum_ = flat.sum(axis=1)
            sumsq = (flat ** 2).sum(axis=1)
        else:
            sum_ += flat.sum(axis=1)
            sumsq += (flat ** 2).sum(axis=1)
        count += flat.shape[1]
        if "mask" in data:
            mask = data["mask"].astype(np.int64)
            labels, freqs = np.unique(mask[mask != 255], return_counts=True)
            for label, freq in zip(labels, freqs):
                class_counts[int(label)] = class_counts.get(int(label), 0) + int(freq)
    mean = (sum_ / count).tolist()
    std = np.sqrt(np.maximum(sumsq / count - np.square(sum_ / count), 1e-8)).tolist()
    return {"mean": mean, "std": std, "pixels": int(count), "class_counts": class_counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--uav-stride", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.uav_stride is None:
        args.uav_stride = args.stride

    random.seed(args.seed)
    np.random.seed(args.seed)
    arcpy.env.overwriteOutput = True

    ensure_dir(args.out, clean=args.clean)
    for sub in ("source_train", "source_val", "target", "uav_val", "tmp"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    source_raster_path = find_first(DATA, "*TOA_32band_rpc_dem_mosaic_clip.tif")
    source_label_path = find_first(DATA, "merge-12-23.shp")
    uav_label_paths = sorted(DATA.rglob("TL0806030*.shp"))
    target_scene_dirs = find_scene_dirs(DATA, "HEM2_20250707002680_")
    if not target_scene_dirs:
        raise FileNotFoundError("No target scenes found")
    if not uav_label_paths:
        raise FileNotFoundError("No UAV label shapefiles found")

    source_raster = arcpy.Raster(str(source_raster_path))
    source_label_raster_path = args.out / "tmp" / "source_label.tif"
    rasterize_polygon_labels(source_label_path, source_raster, source_label_raster_path, name="source")
    source_label_raster = arcpy.Raster(str(source_label_raster_path))

    source_train = export_patches(
        source_raster,
        source_label_raster,
        args.out / "source_train",
        "source_train",
        "train",
        args.patch_size,
        args.stride,
        source_raster_path.stem,
        min_fg_pixels=64,
    )
    source_val = export_patches(
        source_raster,
        source_label_raster,
        args.out / "source_val",
        "source_val",
        "val",
        args.patch_size,
        args.stride,
        source_raster_path.stem,
        min_fg_pixels=0,
    )

    target_infos: list[PatchInfo] = []
    uav_val_infos: list[PatchInfo] = []
    uav_merge = merge_shapefiles(uav_label_paths, args.out / "tmp" / "uav_merge.shp")
    for scene_dir in target_scene_dirs:
        stack_path = args.out / "tmp" / f"{scene_dir.name}_stack.tif"
        stack = arcpy.Raster(str(composite_bands(scene_dir, stack_path)))
        target_infos.extend(
            export_patches(
                stack,
                None,
                args.out / "target",
                f"target_{scene_dir.name}",
                "train",
                args.patch_size,
                args.stride,
                scene_dir.name,
            )
        )
        uav_label_raster_path = args.out / "tmp" / f"{scene_dir.name}_uav_label.tif"
        rasterize_polygon_labels(Path(uav_merge), stack, uav_label_raster_path, name=scene_dir.name)
        uav_label_raster = arcpy.Raster(str(uav_label_raster_path))
        uav_val_infos.extend(
            export_patches(
                stack,
                uav_label_raster,
                args.out / "uav_val",
                f"uav_{scene_dir.name}",
                "all",
                args.patch_size,
                args.uav_stride,
                scene_dir.name,
                min_fg_pixels=32,
            )
        )

    manifest = {
        "source_raster": str(source_raster_path),
        "source_label": str(source_label_path),
        "target_scenes": [str(p) for p in target_scene_dirs],
        "uav_labels": [str(p) for p in uav_label_paths],
        "patch_size": args.patch_size,
        "stride": args.stride,
        "uav_stride": args.uav_stride,
        "seed": args.seed,
        "counts": {
            "source_train": len(source_train),
            "source_val": len(source_val),
            "target": len(target_infos),
            "uav_val": len(uav_val_infos),
        },
        "stats": compute_stats(sorted((args.out / "source_train").glob("*.npz"))),
        "source_train_examples": [asdict(x) for x in source_train[:5]],
        "source_val_examples": [asdict(x) for x in source_val[:5]],
        "uav_val_examples": [asdict(x) for x in uav_val_infos[:5]],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
