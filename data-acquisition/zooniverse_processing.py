from collections import namedtuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import json
import regions

# Required for helioprojective frame
from sunpy import coordinates
import astropy.time as atime
from astropy import coordinates

from astropy import wcs
import astropy.units as u


@dataclass
class ZooniverseExtract:
    bounding_boxes: list[dict[str, float]] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)

    @u.quantity_input()
    def bounding_corners_from_boxes(self) -> (u.arcsec, u.arcsec):
        """Given Zooniverse box and metadata for a given sample,
        extract the (lower left, upper right) corners in arcseconds of the minimum bounding box of the
        volunteer boxes.
        This is to be used with sunpy submaps for making new image crops."""
        corners = list()
        minx, miny, maxx, maxy = (np.inf, np.inf, -np.inf, -np.inf) << u.arcsec
        for box in self.bounding_boxes:
            corners = physical_corners_from_zooniverse(box, self.meta)
            for c in corners:
                minx = min(c[0], minx)
                miny = min(c[1], miny)
                maxx = max(c[0], maxx)
                maxy = max(c[1], maxy)

        lower_left = (minx, miny) << u.arcsec
        upper_right = (maxx, maxy) << u.arcsec
        return (lower_left, upper_right) << u.arcsec


def load_zooniverse_csv(fn: str, cutoff_version: float) -> dict[int, ZooniverseExtract]:
    """Load the Zooniverse CSV export into a dict for further processing."""
    # Keep only the Zooniverse data we care about, and discard the rest
    df = pd.read_csv(fn)

    # Exclude beta testing by restricting the version here
    cut = df["workflow_version"] >= cutoff_version
    df = df[cut]
    ret: dict[int, ZooniverseExtract] = dict()

    for i in range(df.shape[0]):
        row = df.iloc[i]
        id_ = int(row["subject_ids"])
        if id_ not in ret.keys():
            ret[id_] = ZooniverseExtract()

        # The annotations row contains information on the bounding rectangles
        # contained in the current image set under investigation
        boxes = extract_jethunter_annotations(json.loads(row["annotations"]))
        ret[id_].bounding_boxes.extend(boxes)

        # The metadata of the current subject contains things like time,
        # image translation numbers, and FITS headers for physical coordinate conversion.
        # But, each subject ID will be visited by many volunteers, so only save the metadata
        # one time.
        if len(ret[id_].meta.keys()) == 0:
            ret[id_].meta = extract_jethunter_subject_data(
                json.loads(row["subject_data"])
            )

    return ret


def extract_jethunter_annotations(ann: dict[str, object]) -> list[dict[str, float]]:
    """From the `annotations` JSON data entry in a JetHunter export,
    extract the properties of the rectangles that we need to convert to physical coordinates."""
    ret = list()
    for dat in ann:
        # Ignore anything in the annotation data
        # that doesn't have a list of outputs associated with it
        values: list[dict[str, object]] = dat["value"]
        if not isinstance(values, list):
            continue

        for i, v in enumerate(values):
            # Only focus on the rectangle data entries
            if "Rectangle" in v["toolType"]:
                ret.append(extract_bounding_box_params(values, i))
    return ret


def extract_bounding_box_params(
    responses: dict[str, object], rect_idx: int
) -> dict[str, float]:
    # Keys from the rectangle data entries we need to keep
    rect_keep = ("angle", "width", "height", "x_center", "y_center")

    # The current response is the one with the rectangle info
    rect_info = responses[rect_idx]
    # The base point at the start of the event and end of the event
    # are the two data piecces immediately before the rect
    start, end = responses[rect_idx - 2], responses[rect_idx - 1]
    ret = dict()

    # Time since the images started displaying where the event occurs first
    ret["start_time_proportion"] = start["displayTime"]
    ret["end_time_proportion"] = end["displayTime"]
    # Time the user indicated the box is
    ret["box_time_proportion"] = rect_info["displayTime"]
    # Slice out the rectangle info we want
    ret |= {k: rect_info[k] for k in rect_keep}
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

    The extract data is defined [here](https://github.com/kekoalasko/Solar_Zooniverse_Processor/blob/853789e2cd00be82796d6a041b1dd74e039d2d89/solar/visual/img.py#L22-L36)
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


def physical_corners_from_zooniverse(box: dict[str, float], meta: dict[str, object]):
    """Given Zooniverse bounding box data and its associated metadata,
    compute physical coordinates of the box corners in helioprojective coordinates
    and return them."""
    zoon_rect = regions.RectanglePixelRegion(
        regions.PixCoord(box["x_center"], box["y_center"]),
        width=box["width"],
        height=box["height"],
        angle=(np.pi - box["angle"] << u.deg),
    )

    # Convert these corners to physical coordinates
    return tuple(
        zooniverse_coord_to_helioprojective(meta, c << u.pixel)
        for c in zoon_rect.corners
    )


JetHunterRegionTuple = namedtuple("JetHunterRegionTuple", ["region", "time_window"])


def sky_region_from_zooniverse_rect(
    box: dict[str, float], meta: dict[str, object]
) -> JetHunterRegionTuple:
    """
    ## Definition
    Given a set of Zooniverse box data and its associated metadata,
    convert the box coordinates into physical sky coordinates.
    They are sky coordinates observed by AIA across the valid time interval.


    ## Further comments
    Zooniverse uses the upper left corner of each image as the origin.
    The x coordinate increases to the right, and the y coordinate increases downwards.
    The angle definition is [here](https://github.com/zooniverse/front-end-monorepo/blob/beaf46fc9a6316c77f598f5d37f666a52870ea7a/packages/lib-classifier/src/plugins/drawingTools/models/marks/Mark/Mark.js#L58-L61).
    To convert from this left-handed coordinate system to the right-handed one,
    we just need to take the negative value of the angle.
    Zooniverse essentially measures the angle in the counter-clockwise sense from the
    left part of their x axis.
    """
    c1, c2, c3, _ = physical_corners_from_zooniverse(box, meta)

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

    ta, tb = atime.Time((meta["time"]["start_time"], meta["time"]["end_time"]))

    tdelta = tb - ta
    # The jet was only observed between the times
    # specified in the metadata
    start_shift = box["start_time_proportion"] * tdelta
    end_shift = box["end_time_proportion"] * tdelta
    ta, tb = (ta + start_shift), (ta + end_shift)

    # Set the observation time to the spot the box was created
    dt = ta - tb
    loc = box["box_time_proportion"]
    obstime = ta + loc * dt

    return JetHunterRegionTuple(
        regions.RectangleSkyRegion(
            center=coordinates.SkyCoord(
                *physical_center,
                frame="helioprojective",
                observer="earth",
                obstime=obstime,
            ),
            width=physical_width,
            height=physical_height,
            angle=(np.pi - box["angle"] << u.deg),
        ),
        atime.Time((ta, tb)),
    )
