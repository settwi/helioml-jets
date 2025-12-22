"""
Generate an AIA movie given a date that can be parsed by `astropy.time.Time`.

We search the Zooniverse CSV export for a matching time. If one is found,
all frames from that AIA movie get downloaded, and a bunch of PNGs are produced.

You can stitch them together using `ffmpeg` if you want.
"""

from PIL import Image
import numpy as np
import astropy.coordinates as acoord

from sunpy import coordinates  # NOQA
import astropy.time as atime
import pathlib
import parse
import sunpy.map as smap
from sunpy.map.sources.sdo import AIAMap
import sunpy.net.attrs as a
from sunpy.net import Fido
import zooniverse_processing as zp
import sys
import astropy.units as u


def time_to_aia_tai(t: atime.Time) -> str:
    """Generate a DRMS AIA string from an astropy time"""
    parse_str = "{year}-{month}-{day}T{hh}:{mm}:{ss}"
    parsed_aia = parse.parse(parse_str, str(t.tai)).named
    fmt_str = "{year}.{month}.{day}_{hh}:{mm}:{ss}_TAI"
    return fmt_str.format(**parsed_aia)


time = atime.Time(sys.argv[1])
data = zp.load_zooniverse_csv("box-the-jets-first-version.csv", cutoff_version=4.52)

movie_id = None
for id_, d in data.items():
    aia_ranges = sorted(zp.parse_aia_cutout_fn(fn) for fn in d.meta["frame_filenames"])
    ta, tb = aia_ranges[0], aia_ranges[-1]
    if ta < time < tb:
        print("movie for", ta, tb)
        movie_id = id_
        break

if movie_id is None:
    raise ValueError("No valid JetHunter times found.")


extract = data[movie_id]
l, r = extract.bounding_corners_from_boxes()
# Get the (X, Y) extents and find a good amount of padding
(dx, dy) = r - l
pad_pct = 0.2

lower_left, upper_right = (
    acoord.SkyCoord(*c, observer="earth", obstime=ta, frame="helioprojective")
    for c in extract.bounding_corners_from_boxes(padding=pad_pct * max(dx, dy))
)

cutout = a.jsoc.Cutout(lower_left, upper_right)

out_path = pathlib.Path("aia_movie") / str(movie_id)
out_path.mkdir(parents=True, exist_ok=True)

# Now we have the times and the coordinates
# of the different Zooniverse boxes.
# Let's make an AIA movie out of those.

trange = a.Time(ta, tb)
query = Fido.search(
    trange,
    a.Wavelength(304 << u.angstrom),
    a.jsoc.Series.aia_lev1_euv_12s,
    a.jsoc.Notify("wilbert.steinbun@gmail.com"),
    a.jsoc.Segment.image,
    cutout,
)

print(query)

files = Fido.fetch(query, path=out_path / "fits")

(out_path / "pngs").mkdir(exist_ok=True)
frame = 0
sequence = smap.Map(files, sequence=True)
for m in sequence:
    m: AIAMap
    out_scale = m.plot_settings["norm"](m.data)
    max_val = 2**16 - 1
    # Flip the array along the Y axis so the pngs are oriented
    # the same as AIA.
    # And, turn it into a 16-bit grayscale image.
    img = Image.fromarray(np.flip((max_val * out_scale).astype(np.uint16), axis=0))
    img.save(out_path / "pngs" / f"{frame}.png")
    frame += 1

# Create the actual movie, if you want
# ffmpeg -f image2 -framerate 4 -i %d.png -vcodec libx264 -crf 0 test.mp4
