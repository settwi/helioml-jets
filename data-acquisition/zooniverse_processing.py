import numpy as np
import regions

# Required for helioprojective frame
from sunpy import coordinates
import astropy.time as atime
from astropy import coordinates

from astropy import wcs
import astropy.units as u


# TODO update this to export relative times along w/ rectangles
def extract_jethunter_annotations(ann: dict[str, object]) -> list[dict[str, float]]:
    """From the `annotations` JSON data entry in a JetHunter export,
    extract the properties of the rectangles that we need to convert to physical coordinates."""
    # Keys from the rectangle data entries we need to keep
    rect_keep = ("angle", "width", "height", "x_center", "y_center")

    ret = list()
    for dat in ann:
        # Ignore anything in the annotation data
        # that doesn't have a list of outputs associated with it
        if not isinstance(values := dat["value"], list):
            continue

        for v in values:
            # Only focus on the rectangle data entries
            if "Rectangle" not in v["toolType"]:
                continue
            ret.append({k: v[k] for k in rect_keep})
    return ret


def extract_jethunter_subject_data(
    subject_data: dict[str, object],
) -> list[dict[str, object]]:
    """Extract information from the `subject_data` Zooniverse JSON structure to aid in calibrating the
    solar coordinates onto the Zooniverse photo pixel coordinates.

    This includes:
        - FITS header info (presumably exported from AIA images),
        - image padding info (the plot frame around the images)
        - time info (the time of the series of images displayed)

    """
    # Keys required to build up the FITS header for a WCS,
    # presumably taken directly from the AIA fits headers
    fits_header_keys = (
        "#naxis1",
        "#naxis2",
        "#cunit1",
        "#cunit2",
        "#crval1",
        "#crval2",
        "#cdelt1",
        "#cdelt2",
        "#crpix1",
        "#crpix2",
        "#crota2",
    )

    # Keys required to align the image pixels back to
    # FITS pixels
    img_extraction_keys = {
        # The _prop variables are proportions of the total image width or height
        "#im_ll_x": "lower_left_x_prop",
        "#im_ll_y": "lower_left_y_prop",
        "#im_ur_x": "upper_right_x_prop",
        "#im_ur_y": "upper_right_y_prop",
        "#width": "width",
        "#height": "height",
    }

    # Should only be one entry in this piece of data; if not, need to reevaluate
    assert len(keys := list(subject_data.keys())) == 1
    only_key = keys[0]
    sd = subject_data[only_key]


    # "fits_header": {k[1:]: sd[k] for k in fits_header_keys},
    fits_header = dict()
    for k in fits_header_keys:
        try:
            fits_header[k[1:]] = float(sd[k])
        except ValueError:
            fits_header[k[1:]] = sd[k]

    return {
        "fits_header": fits_header,
        "image_extract_data": {
            img_extraction_keys[k]: float(sd[k]) for k in img_extraction_keys
        },
        "time": {
            k[1:]: sd[k].replace(" ", "T") + "Z" for k in ("#start_time", "#end_time")
        },
    }


@u.quantity_input()
def zooniverse_coord_to_helioprojective(
    metadata: dict, original_coordinate: u.pixel
) -> u.arcsec:
    """Given metadata extracted from a Zooniverse object, compute
    the conversion of the Zooniverse pixel coordinate into physical
    helioprojective coordinates.

    The metadata dictionary must contain at least the following:
     - The extract data for each image (i.e. where the solar image is located inside the Zooniverse image)
     - The FITS header metadata from the original AIA image exports.

    These are combined to transform the Zooniverse pixels to AIA pixels,
    then from AIA pixels to physical coordinates.
    """
    # Extract the corner coordinates of the solar sub-section of the Zooniverse image
    # (there are axes surrounding the AIA image)
    extract_data = metadata["image_extract_data"]

    # The lower left location of the subimage in the Zooniverse frame,
    # in native pixels
    lower_left = (
        extract_data["width"] * extract_data["lower_left_x_prop"],
        extract_data["height"] * (1 - extract_data["lower_left_y_prop"]),
    ) << u.pixel

    # The upper right location of the subimage in the Zooniverse frame,
    # in native pixels
    upper_right = (
        extract_data["width"] * extract_data["upper_right_x_prop"],
        extract_data["height"] * (1 - extract_data["upper_right_y_prop"]),
    ) << u.pixel

    # The FITS file image dimensions are different than the Zooniverse images
    fits_header = metadata["fits_header"]
    fits_width = fits_header["naxis1"] << u.pix
    fits_height = fits_header["naxis2"] << u.pix

    # Undo the transformation defined in Paloma Jol's masters thesis
    fits_coord = ([fits_width, fits_height] << u.pix) * (
        (original_coordinate - lower_left) / (upper_right - lower_left)
    )

    # Construct a WCS system using the FITS header information
    system = wcs.WCS(header=fits_header)
    return system.pixel_to_world(*fits_coord) << u.arcsec


def sky_region_from_zooniverse_rect(
    box: dict[str, float], meta: dict[str, object]
) -> regions.RectangleSkyRegion:
    """
    ## Definition
    Given a set of Zooniverse box data and its associated metadata,
    convert the box coordinates into physical sky coordinates.
    They are sky coordinates observed by AIA across the valid time interval.


    ## Further comments
    Zooniverse uses the upper left corner of each image as the origin.
    The x coordinate increases to the right, and the y coordinate increases downwards.
    The angle definition is here: https://github.com/zooniverse/front-end-monorepo/blob/beaf46fc9a6316c77f598f5d37f666a52870ea7a/packages/lib-classifier/src/plugins/drawingTools/models/marks/Mark/Mark.js#L58-L61
    To convert from this left-handed coordinate system to the right-handed one,
    we just need to take the negative value of the angle.
    Zooniverse essentially measures the angle in the counter-clockwise sense from the
    left part of their x axis.
    """
    zoon_rect = regions.RectanglePixelRegion(
        regions.PixCoord(box["x_center"], box["y_center"]),
        width=box["width"],
        height=box["height"],
        angle=-(box["angle"] << u.deg),
    )

    # Convert these corners to physical coordinates
    c1, c2, c3, _ = tuple(
        zooniverse_coord_to_helioprojective(meta, c << u.pixel)
        for c in zoon_rect.corners
    )

    """
    The corners of the `regions` rectangle are defined like this:
    4-------------------3
    |                   |
    |         C         |
    |                   |
    1-------------------2
    where C is the center.
    C is the average of either (1) and (3), or (2) and (4).
    The width is the length between (1) and (2),
    and the height is the length between (2) and (3).
    """
    physical_center = (c1 + c3) / 2
    physical_width = np.hypot(*(c2 - c1))
    physical_height = np.hypot(*(c3 - c2))

    # Set the observation time to the middle of this interval
    ta, tb = atime.Time((meta["time"]["start_time"], meta["time"]["end_time"]))
    obstime = ta + (ta - tb) / 2

    return regions.RectangleSkyRegion(
        center=coordinates.SkyCoord(
            *physical_center, frame="helioprojective", observer="earth", obstime=obstime
        ),
        width=physical_width,
        height=physical_height,
        angle=zoon_rect.angle,
    )
