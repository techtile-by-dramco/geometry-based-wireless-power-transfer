#!/usr/bin/env python3
import uhd
import numpy as np
import time
import os
import sys

# check if UHD_IMAGES_DIR is set, no use in continuing otherwise because failure is guaranteed
if os.environ.get("UHD_IMAGES_DIR"):
    print(os.environ.get("UHD_IMAGES_DIR"))
else:
    print("UHD_IMAGES_DIR not set")
    sys.exit(-1)

# how to use a custom fpga image
script_dir = os.path.dirname(os.path.realpath(__file__))
fpga_path = os.path.join(script_dir, "usrp_b210_fpga_loopback.bin")

########################################
# User parameters
########################################
CENTER_FREQ = 915e6      # Hz
SAMPLE_RATE = 1e6        # Hz
GAIN = 30                # dB
NUM_SAMPS = 100000       # Number of samples to capture
CHANNEL = 0              # RX channel 0 = first channel on B210

########################################
# Create USRP
########################################
usrp = uhd.usrp.MultiUSRP(f"fpga={fpga_path}")

# Set rate, frequency, gain
usrp.set_rx_rate(SAMPLE_RATE, CHANNEL)
usrp.set_rx_freq(CENTER_FREQ, CHANNEL)
usrp.set_rx_gain(GAIN, CHANNEL)

# Give hardware time to lock PLLs
time.sleep(0.1)

########################################
# Create RX streamer
########################################
stream_args = uhd.usrp.StreamArgs("fc32", "sc16")   # Complex float32 samples
rx_streamer = usrp.get_rx_stream(stream_args)

# Buffer for samples
buffer_samps = np.zeros((NUM_SAMPS,), dtype=np.complex64)

########################################
# Issue streaming command
########################################
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
stream_cmd.num_samps = NUM_SAMPS
stream_cmd.stream_now = True
rx_streamer.issue_stream_cmd(stream_cmd)

########################################
# Receive samples
########################################
num_rx = 0
md = uhd.types.RXMetadata()

while num_rx < NUM_SAMPS:
    samps = rx_streamer.recv(buffer_samps[num_rx:], md, timeout=2.0)
    num_rx += samps
    if md.error_code != uhd.types.RXMetadataErrorCode.none:
        print("RX error:", md.strerror())
        break

print(f"Received {num_rx} samples")

########################################
# Compute simple measurement
########################################
# Power (RMS)
rms = np.sqrt(np.mean(np.abs(buffer_samps[:num_rx])**2))
power_db = 20 * np.log10(rms)

with open("power.txt", "w") as f:
    f.write(f"Measured RMS: {rms:.6f}\n")
    f.write(f"Power (dBFS): {power_db:.2f} dBFS\n")

########################################
# Done
########################################
print("Done.")
