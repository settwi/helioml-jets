"""
The jets from SolarJetHunter have times associated with all of them.
Instead of downloading jets using those times, just take the (start, end) times and download
the entire AIA data set.
"""

import pathlib
import sys

import astropy.time as atime
import astropy.units as u
from astropy.io import fits
import asdf
import sunpy.net.attrs as a
from sunpy.net import Fido


def main():
    exports_file = "jet-regions.asdf"
    af = asdf.open(exports_file)
    # For each jet, there is a time range
    # Use it to download an entire set of jet images

    try:
        base_path = pathlib.Path(f"{sys.argv[1]}/data")
    except IndexError:
        base_path = pathlib.Path("./data")

    for jet_id in af.keys():
        print("on id", jet_id)
        if not jet_id.startswith("sjh"):
            continue

        skip = verify_files(base_path / jet_id, af[jet_id]["start_time"], af[jet_id]["end_time"], 12)
        if skip:
            print("skipping", jet_id, "as we have all files")
            continue

        start, end = af[jet_id]["start_time"], af[jet_id]["end_time"]
        time_range = a.Time(start, end)

        email = "wilbert.steinbun@gmail.com"
        query = Fido.search(
            time_range,
            a.Wavelength(304 << u.angstrom),
            a.jsoc.Series.aia_lev1_euv_12s,
            a.jsoc.Notify(email),
            a.jsoc.Segment.image,
        )

        _ = Fido.fetch(query, path=(base_path / jet_id), max_conn=10)


def verify_files(
    path: pathlib.Path, start: atime.Time, end: atime.Time, epsilon: u.s
) -> bool:
    """
    Given a directory full of AIA FITS images, ensure that the earliest and latest
    header observation times are within `epsilon` of `start` and `end`, respectively."""
    try:
        times = times_from_path(path)
    except FileNotFoundError:
        return False
    times = times.sort()
    return ((times[0] - start) <= epsilon) and ((times[-1] - end) <= epsilon)


def times_from_path(path: pathlib.Path) -> atime.Time:
    """Return an array of astropy times from a directory full of
    AIA FITS files."""
    observation_times = list()
    for file in path.iterdir():
        with fits.open(file) as hdus:
            observation_times.append(hdus["COMPRESSED_IMAGE"].header["DATE-OBS"])
    return atime.Time(observation_times)


if __name__ == "__main__":
    main()
