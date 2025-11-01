import json
import os

import astropy.coordinates as acoord
import astropy.time as atime
import astropy.units as u
import numpy as np
import regions
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

    os.makedirs("raw_images", exist_ok=True)
    os.makedirs("normalized_images", exist_ok=True)
    os.makedirs("labels", exist_ok=True)

    # Sort the jet IDs so we get reproducible results
    keys = sorted(box_files.keys())

    jet_id = 0
    for id_ in keys:
        files = box_files[id_]
        pair = extracted[id_]
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
                px: RectanglePixelRegion = box.region.to_pixel(wcs=submap.wcs)

            # Find an un-rotated minimum bounding box
            corners = px.corners
            minx, miny, maxx, maxy = np.inf, np.inf, -np.inf, -np.inf
            for c in corners:
                minx = min(c[0], minx)
                miny = min(c[1], miny)
                maxx = max(c[0], maxx)
                maxy = max(c[1], maxy)
            w = maxx - minx
            h = maxy - miny
            center = regions.PixCoord((minx + maxx) / 2, (miny + maxy) / 2)

            # Compute the width, height, center in
            # normalized formats that YOLO wants
            w, h = px.width, px.height
            npix_y, npix_x = submap.data.shape
            w = w / npix_x
            h = h / npix_y
            c = center.xy / np.array((npix_x, npix_y))

            obst: atime.Time = submap.observer_coordinate.obstime
            year = obst.strftime("%Y")
            base_fn = f"{year}_{jet_id}_{box_id}"

            with open(f"labels/{base_fn}.txt", "w") as f:
                print(f"0 {c[0]:.5f} {c[1]:.5f} {w:.5f} {h:.5f}", file=f)

            raw = submap.data
            raw[raw < 0] = 0
            normalized = submap.plot_settings["norm"](raw)

            max_val = 2**16 - 1
            save_yolo_img(
                (max_val * normalized).astype(np.uint16),
                f"normalized_images/{base_fn}.png",
            )

            # The raw data needs to get scaled
            # appropriately; put it so that it is a
            # linear brightness scale across [0, max val]
            save_yolo_img(
                (max_val * (raw / raw.max())).astype(np.uint16),
                f"raw_images/{base_fn}.png",
            )
            box_id += 1
            print("done box", box_id)

        print("done jet", jet_id)
        jet_id += 1
