from PIL import Image
import regions
from sunpy.coordinates import SphericalScreen
import os
from sunpy import map as smap
from sunpy.map.sources.sdo import AIAMap
import astropy.units as u
import astropy.coordinates as acoord
from regions import RectanglePixelRegion
import zooniverse_processing as zp
import json
import numpy as np


def save_yolo_img(a: np.ndarray, fn: str):
    a = np.array(a / a.max() * 255, dtype=np.uint8)
    img = Image.fromarray(a)
    img.save(fn)


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
cutoff_version = 50.63
extracted = zp.load_zooniverse_csv("box-the-jets.csv", cutoff_version=cutoff_version)

os.makedirs("raw_images", exist_ok=True)
os.makedirs("normalized_images", exist_ok=True)
os.makedirs("labels", exist_ok=True)

i = 0
for id_, files in box_files.items():
    print("start", i)
    pair = extracted[id_]
    # Lower left, upper right bounding corner of all boxes
    # in this Zooniverse event
    lower_left, upper_right = pair.bounding_corners_from_boxes()
    width, height = (upper_right - lower_left) << u.deg
    # Extend the bounding box so it is not a tight crop
    lower_left = lower_left - ((width, height) << u.deg) / 2
    width, height = 2 * width, 2 * height

    boxes = [
        zp.sky_region_from_zooniverse_rect(b, pair.meta) for b in pair.bounding_boxes
    ]

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

            with open(f"labels/{i}.txt", "w") as f:
                print("0 {c[0]:.5f} {c[1]:.5f} {w:.5f} {h:.5f}", file=f)

            normalized = submap.plot_settings["norm"](submap.data)
            raw = submap.data

            save_yolo_img(normalized, f"normalized_images/{i}.png")
            save_yolo_img(raw, f"raw_images/{i}.png")

            print("done", i)
            i += 1
