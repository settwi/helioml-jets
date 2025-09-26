import astropy.time as atime
import pathlib
import zooniverse_processing as zp

# Base directory which contains the solar jet hunter data we care about
base_directory = pathlib.Path("jh-data/")

zoon = zp.load_zooniverse_csv("box-the-jets.csv", cutoff_version=50.63)

for id_, unique_jet_set in zoon.items():
    time_part = unique_jet_set.meta["time"]
    ts, te = atime.Time((time_part["start_time"], time_part["end_time"]))
    print(ts, te)
    # blhe
