"""
Given the Zooniverse exports,
reassociate each volunteer's bounding box with the appropriate AIA image,
and save the output to a large table.
"""

import json
import pathlib
import sys

import astropy.units as u
import zooniverse_processing as zp
import traceback

root = pathlib.Path(sys.argv[1])

# cutoff_version = 50.63
# extracted = zp.load_zooniverse_csv("box-the-jets.csv", cutoff_version=cutoff_version)
# V4.52 is "production"
extracted: dict[int, zp.ZooniverseExtract] = zp.load_zooniverse_csv(
    "box-the-jets-first-version.csv", cutoff_version=4.52
)

# Allowable time shift between frames
# used for cutouts
dt = 2 * (24 << u.s)
output = dict()
keys = sorted(extracted)
for id_ in keys:
    ex = extracted[id_]
    try:
        box_files = zp.reassociate_bounding_boxes(
            ex.meta, ex.bounding_boxes, root, epsilon=dt
        )
        out = list()
        for b in box_files:
            if b is None:
                raise ValueError("None found for best file associated with box")
            out.append(str(b.absolute()))
        output[id_] = out

    except IndexError:
        print("index error for id", id_)
        traceback.print_exc()
    except ValueError as e:
        print("value error:", e.args)
        traceback.print_exc()
    except TypeError as e:
        print("type error:", e.args)
        traceback.print_exc()
    else:
        print("good for id", id_)

out_fn = "box_table.json"
with open(out_fn, "w") as f:
    json.dump(output, f)
