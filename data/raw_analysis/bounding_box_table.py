"""
Given the Zooniverse exports,
reassociate each volunteer's bounding box with the appropriate AIA image,
and save the output to a large table.
"""

import zooniverse_processing as zp
import pathlib
import sys
import json

root = pathlib.Path(sys.argv[1])

cutoff_version = 50.63
extracted = zp.load_zooniverse_csv("box-the-jets.csv", cutoff_version=cutoff_version)

output = dict()
for id_, ex in extracted.items():
    try:
        box_files = zp.reassociate_bounding_boxes(ex.meta, ex.bounding_boxes, root)
        out = list()
        for b in box_files:
            if b is None:
                raise ValueError("None found for best file associated with box")
            out.append(str(b.absolute()))
        output[id_] = out

    except IndexError:
        print("failed for id", id_)
    except ValueError as e:
        print("value error: ", e.args)
    else:
        print("good for id", id_)

out_fn = "box_table.json"
with open(out_fn, "w") as f:
    json.dump(output, f)
