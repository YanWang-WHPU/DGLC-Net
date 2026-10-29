from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from osgeo import gdal


def warp_label_to_stack(stack_path: Path, label_path: Path, out_path: Path) -> None:
    stack = gdal.Open(str(stack_path))
    if stack is None:
        raise FileNotFoundError(stack_path)
    label = gdal.Open(str(label_path))
    if label is None:
        raise FileNotFoundError(label_path)

    gt = stack.GetGeoTransform()
    bounds = (
        gt[0],
        gt[3] + gt[5] * stack.RasterYSize,
        gt[0] + gt[1] * stack.RasterXSize,
        gt[3],
    )
    nodata = label.GetRasterBand(1).GetNoDataValue()
    options = gdal.WarpOptions(
        format="GTiff",
        outputBounds=bounds,
        xRes=abs(gt[1]),
        yRes=abs(gt[5]),
        resampleAlg="near",
        srcNodata=nodata if nodata is not None else 255,
        dstNodata=255,
        outputType=gdal.GDT_Byte,
        creationOptions=["COMPRESS=LZW"],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdal.Warp(str(out_path), str(label_path), options=options)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-uav-dir", type=Path, default=Path("processed_hsi_uda/uav_val"))
    parser.add_argument("--tmp-dir", type=Path, default=Path("processed_hsi_uda/tmp"))
    parser.add_argument("--out-dir", type=Path, default=Path("processed_hsi_uda_aligned/uav_val"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    aligned_labels: dict[str, Path] = {}
    files = sorted(args.source_uav_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No UAV patches found in {args.source_uav_dir}")

    for source in sorted({str(np.load(path, allow_pickle=True)["source"]) for path in files}):
        stack_path = args.tmp_dir / f"{source}_stack.tif"
        label_path = args.tmp_dir / f"{source}_uav_label.tif"
        aligned_path = args.tmp_dir / f"{source}_uav_label_aligned.tif"
        warp_label_to_stack(stack_path, label_path, aligned_path)
        aligned_labels[source] = aligned_path

    for path in files:
        with np.load(path, allow_pickle=True) as data:
            source = str(data["source"])
            x = int(data["x"])
            y = int(data["y"])
            image = data["image"]
        label_ds = gdal.Open(str(aligned_labels[source]))
        mask = label_ds.GetRasterBand(1).ReadAsArray(x, y, image.shape[-1], image.shape[-2]).astype(np.uint8)
        out_path = args.out_dir / path.name
        np.savez_compressed(out_path, image=image, source=source, x=np.int32(x), y=np.int32(y), mask=mask)

    print(f"Wrote {len(files)} aligned UAV patches to {args.out_dir}")


if __name__ == "__main__":
    main()
