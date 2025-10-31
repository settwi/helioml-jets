import datetime
import pathlib
import sys

import astropy.units as u
import sunpy.net.attrs as a
import zooniverse_processing as zp
from sunpy.net import Fido

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

    file_times = dict()

    for id_, extract in data.items():
        # Get the jet start, end bounding times using the AIA cutout file names
        files: list[str] = extract.meta["frame_filenames"]
        files.sort()
        start, end = zp.parse_aia_cutout_fn(files[0]), zp.parse_aia_cutout_fn(files[-1])
        time_range = a.Time(start, end)

        year = start.strftime("%Y")
        if year not in file_times:
            print("getting times for", year)
            file_times[year] = times_from_path(base_path / year)
        skip = verify_times_exist(
            file_times[year],
            start.datetime,
            end.datetime,
            epsilon=datetime.timedelta(seconds=24),
        )
        if skip:
            print("skipping", id_)
            continue

        email = "wilbert.steinbun@gmail.com"
        query = Fido.search(
            time_range,
            a.Wavelength(304 << u.angstrom),
            a.jsoc.Series.aia_lev1_euv_12s,
            a.jsoc.Notify(email),
            a.jsoc.Segment.image,
        )

        # Sort the files into folders by year
        _ = Fido.fetch(query, path=(base_path / year), max_conn=10)


def verify_times_exist(
    times: list[datetime.datetime],
    start: datetime.datetime,
    end: datetime.datetime,
    epsilon: datetime.timedelta,
) -> bool:
    found_start, found_end = False, False
    for t in times:
        if abs(t - start) <= epsilon:
            found_start = True
        if abs(t - end) <= epsilon:
            found_end = True
    return found_start and found_end


def times_from_path(path: pathlib.Path) -> list[datetime.datetime]:
    """Return an array of astropy times from a directory full of
    AIA FITS files."""
    observation_times = list()
    for file in path.iterdir():
        try:
            _, _, time_str, *_ = file.stem.split(".")
            observation_times.append(
                datetime.datetime.strptime(time_str, "%Y-%m-%dT%H%M%SZ")
            )
        except ValueError:
            print(time_str, file.stem)
            raise
    return observation_times


if __name__ == "__main__":
    main()
