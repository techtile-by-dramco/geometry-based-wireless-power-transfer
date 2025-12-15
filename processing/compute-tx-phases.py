import numpy as np
import matplotlib.pyplot as plt
import requests
import yaml
from matplotlib.patches import Rectangle


#################### CONFIGURATIONS ####################
output_path_benchmark = "../client/tx-weights-benchmark.yml"
output_path_friis = "../client/tx-weights-friis.yml"
AMPLTIUDE = 0.8
########################################################


positions_url = r"https://raw.githubusercontent.com/techtile-by-dramco/techtile-description/refs/heads/main/geometry/techtile_antenna_locations.yml"

# Retrieve the file content from the URL
response = requests.get(positions_url, allow_redirects=True)
# Convert bytes to string
content = response.content.decode("utf-8")
# Load the yaml
config = yaml.safe_load(content)

antennas = dict()


# UE position (energy neutral device)
target_location = np.array([3.181, 1.774, 0.266])

# Constants
f = 920e6  # Antenna frequency (Hz)
c = 3e8  # Speed of light (m/s)
lambda_ = c / f  # Wavelength (m)

out_dict = dict()

for c in config["antennes"]:
    # only one antenna is used
    ch1 = c["channels"][1]
    tile_name = c["tile"]
    out_dict[tile_name] = []
    pos = [ch1["x"], ch1["y"], ch1["z"]]
    phase = 0
    ampl = AMPLTIUDE
    d_EN = np.linalg.norm(pos - target_location)  # Scalar distances (L x 1)

    h = np.exp(-1j * 2 * np.pi / lambda_ * d_EN)

    # MRT weights
    w = np.conj(h)
    out_dict[tile_name].append({"ch":1, "ampl": float(AMPLTIUDE), "phase": float(np.rad2deg(np.angle(w)))})
    # CH 0 should be zero in this case
    out_dict[tile_name].append(
        {"ch": 0, "ampl": float(0.0), "phase": float(0.0)}
    )

with open(output_path_friis, "w", encoding="utf-8") as f:
    yaml.safe_dump(out_dict, f, sort_keys=False)
