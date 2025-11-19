import astropy.units as u
import numpy as np
import sunpy.sun.constants as scon

from utils import Jet_box_class as jb
from utils import Jet_class_light as jcl

# We want to restrict the jet centers to be out side of
# at least some portion of the solar angular radius
radius_restriction = 0.99

# For AIA, the angular radius is about 0.25 degrees,
# but we can just calculate it here to be accurate
angular_radius = np.arctan(scon.radius / (1 << u.au)) << u.arcsec

data_file = "catalog.json"
clusters: list[jcl.JetCluster] = jcl.json_import_list(data_file)

unique_ids = set()
for cl in clusters:
    for jet in cl.jets:
        box = jb.Jet_box(
            base=jet.solar_start, height=jet.solar_H, width=jet.solar_W, angle=jet.angle
        )
        c1, _, c3, _ = box.corners()
        center = (c1 + c3) / 2
        center_almost_off_limb = np.hypot(*center) >= (
            radius_restriction * angular_radius
        )
        if center_almost_off_limb:
            unique_ids.add(cl.ID)

with open("limb_events.txt", "w") as f:
    for id_ in unique_ids:
        print(id_, file=f)
