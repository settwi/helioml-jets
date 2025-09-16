import asdf
import copy
from astropy import coordinates as coords
from astropy import units as u
import numpy as np
import os
import regions
from sunpy.coordinates import SphericalScreen
from sunpy import map as smap

"""
Given the Solar Jet Hunter data set which has been translated into a
set of astropy regions exported to an ASDF file,
generate a set of AIA cutout images.

The data is exported to .asdf files.
The jet cutouts are exported in both the default data number pixels,
as well as pixel scaled by the default Sunpy matplotlib normalization.
Each image (jet cutout) gets a pixel array.
Each jet cutout is saved along with the pixel arrays.

The Sunpy-scaled pixels represent human intensity perception better,
and are also in line with prior ML studies of the Solar Jet Hunter products.
However, having the raw pixels could also be useful for training the models.
"""


def main():
    region_fn = "jet-regions.asdf"
    with asdf.open(region_fn) as af:
        # Slice out the data-only keys from the file
        desired_keys = [k for k in af.keys() if k.startswith("sjh")]
        all_regions = dict()

        for k in desired_keys:
            dat = af[k]
            all_regions[k] = {
                "start_time": dat["start_time"],
                "end_time": dat["end_time"],
                "jet_times": dat["jet_times"],
                "regions": [region_from_args(a) for a in dat["jet_regions"]],
            }

    # Next, generate regions which we will use to export
    # new cutout images.
    # Set the numpy seed for repeatable results
    np.random.seed(1337)
    rng = np.random.default_rng()
    cutout_params = generate_cutout_parameters(all_regions, rng=rng)

    # Assume that the images exported by download_all_jethunter_images
    # are within the `data` directory
    image_base_direc = "data"
    available_ids = list(sorted(os.listdir(image_base_direc)))

    asdf_tree = dict()
    for id_ in available_ids:
        # Assume regions are ordered by time in the loaded data product
        # (they should be)
        srt = np.argsort(all_regions[id_]["jet_times"])

        # For each event ID, separate out the bounding box info
        bounding_box_info: list[regions.RectangleSkyRegion] = list(
            np.array(all_regions[id_]["regions"])[srt]
        )

        # Following default AIA image conventions, the file names
        # are lexicographically ordered by time.
        files = os.listdir(f"{image_base_direc}/{id_}")
        aia_fns = list(sorted(files))

        current_cutouts = cutout_params[id_]
        # Each available ID gets a collection of cutouts and corresponding
        # pixel regions.
        cur_tree: dict[str, list] = {
            "raw_cutouts": list(),
            "scaled_cutouts": list(),
            "region_corners": list(),
        }
        for i in range(len(aia_fns)):
            # The AIA file names should sync with the region times,
            # by construction of this data set
            fn = f"{image_base_direc}/{id_}/{aia_fns[i]}"
            map_ = smap.Map(fn)
            (left_corner, right_corner) = current_cutouts[i]
            with SphericalScreen(center=map_.observer_coordinate):
                # Submap the current image with the generated
                # corners of the region
                submap = map_.submap(bottom_left=left_corner, top_right=right_corner)

                # The norm scales the data by human perception; might be more useful
                # to the YOLO or other image recognition algorithms.
                norm = submap.plot_settings["norm"]
                # Export the data into an entry in the ASDF file
                cur_tree["raw_cutouts"].append(raw_dat := submap.data)
                cur_tree["scaled_cutouts"].append(norm(raw_dat).data)

                # Export the corners into pixel coordinates
                box = bounding_box_info[i]
                cur_tree["region_corners"].append(box.to_pixel(wcs=submap).corners)

            # Insert this current chunk into the asdf file tree
            asdf_tree[id_] = cur_tree
            print('finished', id_)

    output_file = asdf.AsdfFile(asdf_tree)
    asdf_fn = "jet_cutouts.asdf"
    output_file.write_to(asdf_fn, all_array_compression="bzp2")


def region_from_args(cur_args: dict[str, object]) -> regions.RectangleSkyRegion:
    """From arguments loaded in from the exported ASDF file of solar jet hunter regions,
    generate a `RectangleSkyRegion`"""
    cur_args = copy.deepcopy(cur_args)
    sky_args = cur_args["skycoord_kwargs"]
    cent = sky_args.pop("center")
    center = coords.SkyCoord(*cent, **sky_args)

    region_args = cur_args["additional_region_kwargs"]
    return regions.RectangleSkyRegion(center, **region_args)


def generate_cutout_parameters(
    region_data: dict[str, object], rng
) -> dict[str, tuple[coords.SkyCoord, coords.SkyCoord]]:
    """Given a dict of jet bounding boxes in physical coordinates,
    generate a set of cutout candidates of (bottom left, top right) corners."""
    cutout_parameters = dict()
    for k in region_data:
        this_data = region_data[k]
        # Each entry stores a list of (bottom left, top right) corners
        cutout_parameters[k] = (corners := list())

        def generate_length(maximal_width):
            """The maximum shift we can have away from a jet encapsulating region is
            determined by its maximal width.
            Otherwise, it would be completely off-screen.
            """
            # The allowable shfit relative to the maximal width
            shift_fraction = 0.5
            return (
                rng.uniform(0, shift_fraction * maximal_width.to_value(u.arcsec))
                << u.arcsec
            )

        for reg in this_data["regions"]:
            cutout_side_length = side_length_from_region(reg)
            dx = generate_length(cutout_side_length)
            dy = generate_length(cutout_side_length)
            bottom_left = coords.SkyCoord(
                reg.center.Tx + dx - cutout_side_length / 2,
                reg.center.Ty + dy - cutout_side_length / 2,
                frame=reg.center.frame,
            )
            top_right = coords.SkyCoord(
                reg.center.Tx + dx + cutout_side_length / 2,
                reg.center.Ty + dy + cutout_side_length / 2,
                frame=reg.center.frame,
            )
            corners.append((bottom_left, top_right))

    return cutout_parameters


@u.quantity_input()
def side_length_from_region(
    region: regions.RectangleSkyRegion, aia_frac: float = 0.2
) -> u.arcsec:
    """Generate a random cutout length from the minimum (region maximal extent)
    to the maximum (fraction of the AIA field of view).

    You can specify the AIA FoV fraction (`aia_frac`) if you want a larger/smaller
    upper bound on the cutout size
    """
    # Add space around the region so we don't end up
    # cutting out too many complete jets
    buffer = 0.2
    min_allowed_side = (1 + buffer) * np.sqrt(region.width**2 + region.height**2)

    full_aia_view = 41 << u.arcmin
    # Max FOV is some fraction of the AIA FOV
    max_allowed_side = (full_aia_view * aia_frac).to(u.arcsec)

    # Restrict the minimum if it's larger than the maximum already
    min_allowed_side = min(min_allowed_side, max_allowed_side)

    return (
        np.random.default_rng().uniform(
            min_allowed_side.to_value(u.arcsec), max_allowed_side.to_value(u.arcsec)
        )
        << u.arcsec
    )


if __name__ == "__main__":
    main()
