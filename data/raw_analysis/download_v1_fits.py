import pathlib
import sys

import astropy.units as u
import sunpy.net.attrs as a
from sunpy.net import Fido

import zooniverse_processing as zp

"""From the JetHunter V1 data set, download the associated AIA fits files into per-year directories"""


def main():
    try:
        base_path = pathlib.Path(f"{sys.argv[1]}")
    except IndexError:
        raise ValueError("Specify the directory to save data into via argv")

    # Exported with panoptes
    first_version_fn = "box-the-jets-first-version.csv"
    # V4.52 is "production"
    data: dict[int, zp.ZooniverseExtract] = zp.load_zooniverse_csv(
        first_version_fn, cutoff_version=4.52
    )

    for extract in data.values():
        # Get the jet start, end bounding times using the AIA cutout file names
        files: list[str] = extract.meta["frame_filenames"]
        files.sort()
        start, end = zp.parse_aia_cutout_fn(files[0]), zp.parse_aia_cutout_fn(files[-1])
        time_range = a.Time(start, end)

        email = "wilbert.steinbun@gmail.com"
        query = Fido.search(
            time_range,
            a.Wavelength(304 << u.angstrom),
            a.jsoc.Series.aia_lev1_euv_12s,
            a.jsoc.Notify(email),
            a.jsoc.Segment.image,
            a.Sample(24 << u.s),
        )

        # Sort the files into folders by year
        _ = Fido.fetch(query, path=(base_path / start.strftime("%Y")), max_conn=10)


if __name__ == "__main__":
    main()
