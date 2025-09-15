from utils import Jet_class_light as jcl
from utils import Jet_box_class as jb

import asdf
import numpy as np
import astropy.units as u
import astropy.time as atime

"""
Convert Sophie's jet box classes into data (constructor arguments)
which can be used to generate astropy regions of the jet boxes.
"""


def main():
    # Load Sophie's format
    data_file = "jet_clusters.json"
    clusters = jcl.json_import_list(data_file)

    # Condense the jet object information down to its essence
    condensed = dict()

    flats = 0
    rots = 0
    for cl in clusters:
        # We need to know the start and end times for later
        # extraction of AIA cutouts
        start_time = atime.Time(cl.obs_time)
        end_time = start_time + (cl.Duration << u.min)
        jet_times = list()

        # Each cluster has a series of jet boxes in it
        condensed[cl.ID] = {"jet_regions": (cur_regions := list())}
        for jet in cl.jets:
            jet: jcl.Jet
            rect_region = astropy_region_from_box(jet, atime.Time(jet.time))
            no_rotation = (
                rect_region["additional_region_kwargs"]["angle"].to_value(u.rad) < 1e-10
            )
            if no_rotation:
                flats += 1
            else:
                rots += 1

            cur_regions.append(rect_region)
            jet_times.append(atime.Time(jet.time))

        # Add additional metadata
        condensed[cl.ID] |= {
            "start_time": start_time,
            "end_time": end_time,
            "jet_times": jet_times,
        }

    print("we had", flats, "flat boxes (unrotated?)")
    print("we had", rots, "rotated boxes")
    # Write the data out into a compressed ASDF file
    af = asdf.AsdfFile()
    af.tree = condensed
    af.write_to(fn := "jet-regions.asdf", all_array_compression="bzp2")
    print(f"Exported {fn}")


def astropy_region_from_box(jet: jcl.Jet, obstime: atime.Time) -> dict[str, object]:
    """Given a `Jet` object from Sophie's code and an `astropy.time.Time` corresponding to the observation,
    generate the data required for a `regions.RectangleSkyRegion`.

    This can be used later to select other sky regions from AIA or other instruments.
    It is a standard, physically-meaningful representation of the jet boxes.
    """
    box = jb.Jet_box(
        base=jet.solar_start, height=jet.solar_H, width=jet.solar_W, angle=jet.angle
    )

    # Transform the box into an astropy region
    c1, c2, c3, _ = box.corners()
    center = (c1 + c3) / 2
    # Put the correct time into the center

    # The sky region needs more info
    (dx, dy) = c2 - c1
    angle = np.atan2(dy, dx)
    return {
        "skycoord_kwargs": {
            "center": center,
            "frame": "helioprojective",
            "observer": "earth",
            "obstime": obstime,
        },
        "additional_region_kwargs": {
            # The astropy region expects width/height to be flipped
            # with respect to this angle
            "width": box.height.to(u.deg),
            "height": box.width.to(u.deg),
            "angle": angle,
        },
    }


if __name__ == "__main__":
    main()
