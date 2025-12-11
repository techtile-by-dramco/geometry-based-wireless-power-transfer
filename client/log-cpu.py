#!/usr/bin/env python3
import psutil
import struct
import time

# Output file
OUT_FILE = "cpu_log.bin"

# Struct format:
#   float32 timestamp
#   float32 cpu_percent
# Total = 8 bytes per sample
record_struct = struct.Struct("ff")

def log_cpu(interval=1.0):
    """
    Logs CPU usage to a binary file.
    Each record: [float32 timestamp, float32 cpu_percent]
    """

    with open(OUT_FILE, "wb") as f:
        print(f"Logging CPU usage every {interval}s → {OUT_FILE}")

        # Warm up psutil (first call always returns 0.0)
        psutil.cpu_percent(interval=None)

        while True:
            timestamp = time.time()
            cpu = psutil.cpu_percent(interval=interval)

            # Pack two float32 values
            data = record_struct.pack(
                float(timestamp),
                float(cpu)
            )

            f.write(data)
            f.flush()

            print(f"{timestamp:.2f}  CPU={cpu:.1f}%")

if __name__ == "__main__":
    log_cpu(interval=1.0)
