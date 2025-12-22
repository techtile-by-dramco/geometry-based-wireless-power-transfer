import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# === Huidig pad opvragen en één niveau hoger ===
current_file_path = os.path.abspath(__file__) 
current_dir = os.path.dirname(current_file_path)

def wrap_phase(x):
    return (x + 180) % 360 - 180

# CSV-bestand inlezen
df = pd.read_csv(f"{current_dir}/results.csv")

# Referentie: CH1 = 0 (faseverschillen t.o.v. CH1)
phase_diff_CH2_CH1 = wrap_phase(df["CH2"])
phase_diff_CH3_CH1 = wrap_phase(df["CH3"])
phase_diff_CH4_CH1 = wrap_phase(df["CH4"])
phase_diff_CH4_CH3 = wrap_phase(df["CH4"]-df["CH3"])
phase_diff_CH3_CH2 = wrap_phase(df["CH3"]-df["CH2"])
phase_diff_CH2_CH4 = wrap_phase(df["CH2"]-df["CH4"])

all_phase_diffs = np.concatenate([
    phase_diff_CH2_CH1.to_numpy(),
    phase_diff_CH3_CH1.to_numpy(),
    phase_diff_CH4_CH1.to_numpy(),
    phase_diff_CH4_CH3.to_numpy(),
    phase_diff_CH3_CH2.to_numpy(),
    phase_diff_CH2_CH4.to_numpy()
])

print(f"Aantal metingen: {len(all_phase_diffs)}")
# print(all_phase_diffs)


# Standaardafwijking berekenen
# std_CH2 = np.std(phase_diff_CH2, ddof=1)
# std_CH3 = np.std(phase_diff_CH3, ddof=1)
# std_CH4 = np.std(phase_diff_CH4, ddof=1)

std = np.std(all_phase_diffs, ddof=1)

print(f"Std (deg): {std:.3f}°")

plt.figure()
plt.plot(all_phase_diffs, marker='o', linestyle='None')
plt.axhline(0)
plt.axhline(std)
plt.axhline(-std)
plt.xlabel("Sample index")
plt.ylabel("Phase difference (deg)")
plt.title(f"All phase differences with ±1σ (σ = {std:.3f}°)")
plt.show()