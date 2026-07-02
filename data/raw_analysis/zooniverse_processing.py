import json
import pathlib
from dataclasses import dataclass, field

# Required for helioprojective frame
from sunpy import coordinates

import astropy.time as atime
import astropy.units as u
import numpy as np
import pandas as pd
import regions
from astropy import coordinates, wcs
from astropy.io import fits


@dataclass
class ZooniverseExtract:
    bounding_boxes: list[dict[str, float]] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)

    @u.quantity_input
    def bounding_corners_from_boxes(
        self, padding: u.arcsec = (0 << u.arcsec)
    ) -> (u.arcsec, u.arcsec):
        """Given Zooniverse box and metadata for a given sample,
        extract the (lower left, upper right) corners in arcseconds of the minimum bounding box of the
        volunteer boxes.
        This is to be used with sunpy submaps for making new image crops.

        The bounding corners are not rotated."""
        corners = list()
        minx, miny, maxx, maxy = (np.inf, np.inf, -np.inf, -np.inf) << u.arcsec
        for box in self.bounding_boxes:
            corners = physical_corners_from_zooniverse(box, self.meta)
            for c in corners:
                minx = min(c[0], minx)
                miny = min(c[1], miny)
                maxx = max(c[0], maxx)
                maxy = max(c[1], maxy)

        lower_left = (minx - padding, miny - padding) << u.arcsec
        upper_right = (maxx + padding, maxy + padding) << u.arcsec
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
        if not ret[id_].meta:
            ret[id_].meta = extract_jethunter_subject_data(
                sd := json.loads(row["subject_data"])
            )

            # Also export the image names for frame-based time reconstruction.
            # This is the only way to do it in the old version, but might be useful
            # in the new version, too.
            only_key = next(iter(sd))
            sd = sd[only_key]
            fns = [v for (k, v) in sd.items() if "file_name" in k]
            ret[id_].meta["frame_filenames"] = fns

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
        ret.append(extract_bounding_box_params(values))

    return ret


def extract_bounding_box_params(responses: dict[str, object]) -> dict[str, float]:
    for i, v in enumerate(responses):
        # Only focus on the rectangle data entries
        if "toolType" in v:
            # "New" version
            if "Rectangle" in v["toolType"]:
                return extract_bounding_box_params_new_version(responses, i)

        if "tool" in v:
            # "Old" version
            if v["tool"] == 2:
                return extract_bounding_box_params_old_version(responses, i)


def extract_bounding_box_params_old_version(
    responses: dict[str, object], rect_idx: int
) -> dict[str, float]:
    rect_keep = ("angle", "width", "height", "x", "y")
    ret = {k: responses[rect_idx][k] for k in rect_keep}

    # The base point at the start of the event and end of the event
    # are the two data piecces immediately before the rect
    start, end = responses[rect_idx - 2], responses[rect_idx - 1]

    # The jet was indicated to start/end at specific frames in each movie sequence.
    # We can use the frames to pick out file names and thus times at which the events occurred.
    ret["start_frame"] = start["frame"]
    ret["end_frame"] = end["frame"]
    # The user indicated that the box was brightest at this frame
    ret["box_time_frame"] = responses[rect_idx]["frame"]
    return ret


def extract_bounding_box_params_new_version(
    responses: dict[str, object], rect_idx: int
) -> dict[str, float]:
    rect_keep = ("angle", "width", "height", "x_center", "y_center")
    ret = {k: responses[rect_idx][k] for k in rect_keep}

    # The base point at the start of the event and end of the event
    # are the two data piecces immediately before the rect
    start, end = responses[rect_idx - 2], responses[rect_idx - 1]

    # At some point, Zooniverse stopped reporting frames and switched to relative times (?)
    try:
        ret["start_frame"] = start["frame"]
        ret["end_frame"] = end["frame"]
        ret["box_time_frame"] = responses[rect_idx]["frame"]
    except KeyError:
        ret["start_time_proportion"] = start["displayTime"]
        ret["end_time_proportion"] = end["displayTime"]
        ret["box_time_proportion"] = responses[rect_idx]["displayTime"]
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

    fits_header = dict()
    # New version is always like this
    if all(k in sd for k in fits_header_keys):
        extract_from = sd
    # Sometimes, the old version only has this data contained in per-frame
    # information clumps (???)
    else:
        # Take the first header and extract what we need,
        # and put the hashtag back at the start of the keys to match
        # the second version of the data set (wtf)
        extract_from = {
            f"#{k}": v for (k, v) in json.loads(sd["#fits_header_0"]).items()
        }

    for k in fits_header_keys:
        try:
            fits_header[k[1:]] = float(extract_from[k])
        except ValueError:
            fits_header[k[1:]] = extract_from[k]

    ret = {
        "fits_header": fits_header,
        "image_extract_data": {
            img_extraction_keys[k]: float(sd[k]) for k in img_extraction_keys
        },
    }

    if "#start_time" in sd:
        ret["time"] = {
            k[1:]: sd[k].replace(" ", "T") + "Z" for k in ("#start_time", "#end_time")
        }

    return ret


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
    # in Kekoa coordinate system
    lower_left = (
        (orig_width := extract_data["width"]) * extract_data["lower_left_x_prop"],
        (orig_height := extract_data["height"]) * extract_data["lower_left_y_prop"],
    ) << u.pixel

    # The upper right location of the subimage in the Zooniverse frame,
    # in Kekoa coordinate system
    upper_right = (
        orig_width * extract_data["upper_right_x_prop"],
        orig_height * extract_data["upper_right_y_prop"],
    ) << u.pixel

    # The solar portion of the image in the entire Zooniverse image is
    # limited by the parameters from the metadata
    (zoon_img_width, zoon_img_height) = upper_right - lower_left

    # Convert the raw coordinate to the one within the image bounds.
    # The "Kekoa" system orients (0, 0) at the lower-left corner, but the
    # Zooniverse system defines it at the upper-left corner of the image.
    kekoa_coord = (
        original_coordinate[0],
        (orig_height << u.pix) - original_coordinate[1],
    ) << u.pix
    (image_x, image_y) = kekoa_coord - lower_left

    # The FITS file image dimensions are different than the Zooniverse images
    fits_header = metadata["fits_header"]
    fits_width = fits_header["naxis1"] << u.pixel
    fits_height = fits_header["naxis2"] << u.pixel

    # Undo the transformation defined in Paloma Jol's masters thesis.
    # The conversion between FITS and "solar" aka Zooniverse perceived
    # widths occurs because of the difference in DPI of presented vs. science
    # images.
    fits_coord = (
        image_x * (fits_width / zoon_img_width),
        image_y * (fits_height / zoon_img_height),
    ) << u.pixel

    # Construct a WCS system using the FITS header information
    system = wcs.WCS(header=fits_header)
    return system.pixel_to_world(*fits_coord) << u.arcsec


def physical_corners_from_zooniverse(box: dict[str, float], meta: dict[str, object]):
    """Given Zooniverse bounding box data and its associated metadata,
    compute physical coordinates of the box corners in helioprojective coordinates
    and return them."""
    # Center position keys are different between Zooniverse versions.
    # Go figure...
    if "x_center" in box:
        xk, yk = "x_center", "y_center"
        center = regions.PixCoord(box[xk], box[yk])
    else:
        # We only have one of the corners defined (upper left?)
        # so we need to "cast" it to the center.
        # This follows the convention from https://github.com/kapsiak/Solar_Zooniverse_Processor/blob/ff006354819e62b586a272a676d2f74479ec66b1/solar/zooniverse/zimport.py#L155-L157
        # The old Zooniverse front end has the box defined from a corner: https://github.com/zooniverse/Panoptes-Front-End/blob/bd99eb6bdcc999ddeca73c7335f904c7e1be69dd/app/classifier/drawing-tools/rotate-rectangle.jsx#L84-L87
        left_x, left_y = box["x"], box["y"]
        w, h = box["width"], box["height"]
        center = regions.PixCoord(
            left_x + w / 2,
            left_y + h / 2,
        )

    zoon_rect = regions.RectanglePixelRegion(
        center,
        width=box["width"],
        height=box["height"],
        # The angle definition from Zooniverse is phase shifted from what
        # astropy regions expects.
        angle=(np.pi - (box["angle"] << u.deg).to_value(u.rad)) << u.rad,
    )

    # Convert these corners to physical coordinates
    return tuple(
        zooniverse_coord_to_helioprojective(meta, c << u.pixel)
        for c in zoon_rect.corners
    )


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

    if "time" in meta:
        ta, tb = atime.Time((meta["time"]["start_time"], meta["time"]["end_time"]))
        loc = box["box_time_proportion"]
        tdelta = tb - ta
        obstime = ta + loc * tdelta
    else:
        # We're on V1
        fns = meta["frame_filenames"]
        idx = box["box_time_frame"]
        obstime = parse_aia_cutout_fn(fns[idx])

    return regions.RectangleSkyRegion(
        center=coordinates.SkyCoord(
            *physical_center,
            frame="helioprojective",
            observer="earth",
            obstime=obstime,
        ),
        width=physical_width,
        height=physical_height,
        angle=(box["angle"] << u.deg),
    )


def parse_aia_cutout_fn(fn: str) -> atime.Time:
    _, _, ymd, hms, *_ = fn.split("_")
    return atime.Time.strptime(
        time_string=f"{ymd}-{hms}", format_string="%Y%m%d-%H%M%S"
    )


def reassociate_bounding_boxes(
    meta: dict[str, object],
    bounding_boxes: list[dict[str, float | int]],
    root_path: pathlib.Path,
    epsilon: u.Quantity,
) -> list[pathlib.Path]:
    """
    Re-associate bounding boxes with particular AIA files in a movie sequence.
    A movie sequence is a sequence of image files, and some of those files have
    bounding boxes drawn on them.

    Custom cutouts may be made out of the movie sequence files assuming the FITS are
    available for manipulation

    Operates on: list of bounding boxes, set of metadata,
    for a jet hunter event.
    """

    if "time" in meta:
        # "New" version stores time information in metadata
        start, end = (
            bounding_times := atime.Time(
                (meta["time"]["start_time"], meta["time"]["end_time"])
            )
        )
    else:
        # "Old" version just stores the file names.
        # We can get the time range via file names
        (fns := meta["frame_filenames"]).sort()
        start_fn, end_fn = fns[0], fns[-1]
        start, end = (
            bounding_times := atime.Time(
                (parse_aia_cutout_fn(start_fn), parse_aia_cutout_fn(end_fn))
            )
        )

    # Assumes files are sorted in directories by year with default AIA naming convention
    first_glob, second_glob = bounding_times.strftime(
        "%Y/aia.lev1_euv_12s.%Y-%m-%dT%H%M*.fits"
    )

    p = pathlib.Path(root_path)
    all_files = tuple(sorted((p / str(start.datetime.year)).iterdir()))

    first_file = next(p.glob(first_glob))
    last_file = tuple(p.glob(second_glob))[-1]

    ai, bi = all_files.index(first_file), all_files.index(last_file)
    file_slice = all_files[ai:bi]

    times = list()
    for file in file_slice:
        with fits.open(file) as f:
            times.append(atime.Time(f[1].header["DATE-OBS"]))

    # The total movie duration in seconds
    dt = (end - start).to(u.s)

    epsilon = epsilon.to_value(u.s)
    box_files = list()
    for bb in bounding_boxes:
        # Old version uses frames to dictate time
        if "box_time_frame" in bb:
            box_time = parse_aia_cutout_fn(
                meta["frame_filenames"][bb["box_time_frame"]]
            )
        else:
            box_time = start + (bb["box_time_proportion"] * dt)
        min_comparison = float("inf")
        best = None
        for i, t in enumerate(times):
            comp = (box_time - t).to_value(u.s)
            if abs(comp) < min_comparison:
                min_comparison = comp
                best = file_slice[i]
        if min_comparison > epsilon:
            raise ValueError(
                f"Couldn't find image within {epsilon:.1f}s of requested time(s)"
            )
        box_files.append(best)

    return box_files
