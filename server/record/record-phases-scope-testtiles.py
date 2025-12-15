import time
import pyvisa as visa
import numpy as np
from scipy.signal import find_peaks


class Scope:
    def __init__(self, ip: str):
        self.ip = ip
        self.rm = visa.ResourceManager()
        self.scope = self.rm.open_resource(f"TCPIP::{ip}::INSTR")
        self.channels = ["CH1", "CH2", "CH3", "CH4"]


    # ------------------------------------------------------------
    # BASIC INITIALIZATION
    # ------------------------------------------------------------
    def _init_scope(self):
        """Reset scope and configure channels."""
        print("[i] Resetting scope…")
        self.scope.write("*rst")

        # Add 3 phase measurements
        print("[i] Adding phase measurements…")
        for _ in range(3):
            self.scope.write("MEASUREMENT:ADDMEAS PHASE")

        # Get measurement names
        self.meas_list = self.scope.query("MEASUrement:LIST?").strip().split(",")

        # Configure channels
        print("[i] Configuring channels…")
        for ch in self.channels:
            self.scope.write(f"{ch}:TERMINATION 50")
            self.scope.write(f"{ch}:BANdwidth 2e9")
            self.scope.write(f"SELECT:{ch} 1")

        # Horizontal scale, overlay view
        self.scope.write("HORIZONTAL:MODE:SCALE 400e-12")
        self.scope.write("DISplay:WAVEView1:VIEWStyle OVERLAY")

        # Link measurements to channels
        print("[i] Mapping measurement sources…")
        for i in range(3):
            self.scope.write(f"MEASUrement:{self.meas_list[i]}:SOUrce1 {self.channels[i+1]}")
            self.scope.write(f"MEASUrement:{self.meas_list[i]}:SOUrce2 CH1")
    
    def check_status(self):
        no_meas = len(self.scope.query("MEASUrement:LIST?").strip().split(","))
        if(no_meas != 3):
            return True
        return False

    # ------------------------------------------------------------
    # READOUT
    # ------------------------------------------------------------
    def read_measurements(self):
        """Read the mean history value for all configured measurements."""
        measurements = self.scope.query("MEASUrement:LIST?").strip().split(",")
        results = {}

        for meas in measurements:
            value = self.scope.query(f"MEASUrement:{meas}:RESUlts:CURRent:MEAN?")
            results[meas] = float(value)
        return results

    # ------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------
    def close(self):
        """Close VISA session."""
        self.scope.close()
        self.rm.close()
        print("[i] VISA session closed.")


# ------------------------------------------------------------
# USAGE
# ------------------------------------------------------------
import os
from datetime import datetime
import csv
if __name__ == "__main__":
    # === Maak Scope object aan en lees data ===
    scope_obj = Scope(ip="192.108.1.219")

    # === Initialize scope
    if scope_obj.check_status():
        scope_obj._init_scope()
    else:
        print("Scope setup already ok!")

    # === Get data
    data = scope_obj.read_measurements()

    # === Huidig pad opvragen en één niveau hoger ===
    current_file_path = os.path.abspath(__file__) 
    current_dir = os.path.dirname(current_file_path)
    parent_path = os.path.dirname(current_dir)
    data_folder = os.path.join(parent_path, "../data")

    # === Zorg dat de folder bestaat ===
    os.makedirs(data_folder, exist_ok=True)

    # === CSV-bestand met timestamp in de naam ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(data_folder, f"testtiles-phase-results.csv")

    # === Schrijf data naar CSV ===
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ts", "CH2", "CH3", "CH4"])  # header
        writer.writerow([timestamp, data["MEAS1"], data["MEAS2"], data["MEAS3"]])

    print(f"Data opgeslagen in {csv_file}")

    scope_obj.scope.close()


