import astropy.units as u
import asdf
import sunpy.net.attrs as a
from sunpy.net import Fido


def main():
    # Load in the pre-saved region data
    with asdf.open("jet-regions.asdf", lazy_load=True) as af:
        # Separate the ASDF builtin keys from the event ID keys
        desired_keys = [k for k in af.keys() if k.startswith("sjh")]
        all_regions = dict()

        for k in desired_keys:
            dat = af[k]
            # Put time data and astropy regions into a dict
            # All we need here are the time windows;
            # region data is ignored
            all_regions[k] = {
                "start_time": dat["start_time"],
                "end_time": dat["end_time"],
                "jet_times": dat["jet_times"],
            }
    print("region data loaded")

    # Keys can change order each iteration, so
    # order them now
    ids = list(sorted(all_regions.keys()))

    times = dict()
    for region_id in ids:
        times[region_id] = (cur_times := list())
        for t in all_regions[region_id]["jet_times"]:
            # Get only the frame immediately associated with the jet
            cur_times.append((t, t + (1 << u.s)))

    print("times sliced up")
    email = "wilbert.steinbun@gmail.com"
    for id in ids:
        print("on id", id)
        cur_times = times[id]
        ta, tb = cur_times[0]
        t_query = a.Time(ta, tb)
        # Combine the time query into a large request
        # (hopefully faster for JSOC to provide it this way)
        for ta, tb in cur_times[1:]:
            t_query = t_query | a.Time(ta, tb)

        query = Fido.search(
            t_query,
            a.Wavelength(304 << u.angstrom),
            a.jsoc.Series.aia_lev1_euv_12s,
            a.jsoc.Notify(email),
            a.jsoc.Segment.image,
        )

        # Sort the files by jet event into different directories.
        # They will default to sorted by time.
        Fido.fetch(query, path=f"data/{id}/")


if __name__ == "__main__":
    main()
