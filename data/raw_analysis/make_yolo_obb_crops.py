import itertools
import json
import multiprocessing as mp
import os
import pathlib

import astropy.coordinates as acoord
import astropy.time as atime
import astropy.units as u
import numpy as np
import zooniverse_processing as zp
from PIL import Image
from regions import RectanglePixelRegion
from sunpy import map as smap
from sunpy.coordinates import SphericalScreen
from sunpy.map.sources.sdo import AIAMap


def save_yolo_img(a: np.ndarray, fn: str):
    img = Image.fromarray(a)
    img.save(fn)


if __name__ == "__main__":
    # Prerequesite: run bounding_box_table.py first
    table_fn = "box_table.json"
    with open(table_fn, "r") as f:
        box_files = json.load(f)

    # Turn all the keys into integers
    for k in tuple(box_files.keys()):
        box_files[int(k)] = box_files[k]
        del box_files[k]

    # Prerequesite: export and download the Zooniverse .csv
    # using panoptes CLI
    # cutoff_version = 50.63
    # extracted = zp.load_zooniverse_csv(
    # "box-the-jets.csv", cutoff_version=cutoff_version
    # )
    # Exported with panoptes
    first_version_fn = "box-the-jets-first-version.csv"
    # V4.52 is "production"
    extracted: dict[int, zp.ZooniverseExtract] = zp.load_zooniverse_csv(
        first_version_fn, cutoff_version=4.52
    )

    label_path = pathlib.Path("labels")
    label_path.mkdir(exist_ok=True)
    norm_path = pathlib.Path("normalized_images")
    norm_path.mkdir(exist_ok=True)

    # Sort the jet IDs so we get reproducible results
    keys = sorted(box_files.keys())

    # Put the box exporting into a function for multiprocessing
    def process_jets(kz):
        for jet_id in kz:
            files = box_files[jet_id]
            pair = extracted[jet_id]
            # Lower left, upper right bounding corner of all boxes
            # in this Zooniverse event
            lower_left, upper_right = pair.bounding_corners_from_boxes()
            width, height = (upper_right - lower_left) << u.deg
            # Extend the bounding box so it is not a tight crop
            lower_left = lower_left - ((width, height) << u.deg) / 2
            width, height = 2 * width, 2 * height

            boxes = [
                zp.sky_region_from_zooniverse_rect(b, pair.meta)
                for b in pair.bounding_boxes
            ]

            box_id = 0
            for box, aia_fn in zip(boxes, files):
                m: AIAMap = smap.Map(aia_fn)
                with SphericalScreen(center=m.observer_coordinate):
                    submap = m.submap(
                        acoord.SkyCoord(*lower_left, frame=m.coordinate_frame),
                        width=width,
                        height=height,
                    )
                    px: RectanglePixelRegion = box.to_pixel(wcs=submap.wcs)

                obst: atime.Time = submap.observer_coordinate.obstime
                date_str = obst.strftime("%Y-%m-%dT%H-%M-%S")
                base_fn = f"{date_str}_{jet_id}_{box_id}"

                # Compute the width, height, center in
                # normalized formats that YOLO wants
                npix_y, npix_x = submap.data.shape
                normalized_corners = [
                    (x / npix_x, y / npix_y)
                    for (x, y) in px.corners
                ]

                with open(label_path / f"{base_fn}.txt", "w") as f:
                    print("0", file=f, end=" ")
                    for (x, y) in normalized_corners:
                        print(f"{x:.4f} {y:.4f}", end=" ", file=f)
                    print(file=f)

                # Export map as 16-bit grayscale
                normalized = submap.plot_settings["norm"](submap.data)
                max_val = 2**16 - 1
                save_yolo_img(
                    (max_val * normalized).astype(np.uint16),
                    norm_path / f"{base_fn}.png",
                )
                print("done box", box_id)
                box_id += 1
            print("done jet", jet_id)

    chunked_keys = np.array_split(keys, num_cpu := os.cpu_count())
    with mp.Pool(num_cpu) as p:
        p.map(process_jets, chunked_keys)
