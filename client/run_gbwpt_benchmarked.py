from utils.client_com import Client
import signal
import time
import sys
import argparse, shlex
from datetime import datetime, timezone, timedelta
import uhd
import numpy as np
import yaml
import logging
import os
import queue
import threading

# =============================================================================
#                           Custom Log Formatter
# =============================================================================
# This formatter adds timestamps with fractional seconds to log messages,
# allowing for more precise event timing (useful in measurement systems).
# =============================================================================

class LogFormatter(logging.Formatter):
    """Custom log formatter that prints timestamps with fractional seconds."""

    @staticmethod
    def pp_now():
        """Return the current time of day as a formatted string with milliseconds."""
        now = datetime.now()
        return "{:%H:%M}:{:05.2f}".format(now, now.second + now.microsecond / 1e6)

    def formatTime(self, record, datefmt=None):
        """Override the default time formatter to include fractional seconds."""
        converter = self.converter(record.created)
        if datefmt:
            formatted_date = converter.strftime(datefmt)
        else:
            formatted_date = LogFormatter.pp_now()
        return formatted_date

# =============================================================================
#                           Logger and Channel Configuration
# =============================================================================
# This section initializes the global logger and defines the
# channel mapping used for reference and loopback measurements.
# =============================================================================

global logger
global begin_time

connected_to_server = False
begin_time = 2.0

# -------------------------------------------------------------------------
# Logger setup
# -------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Stream logs to console
console = logging.StreamHandler()
logger.addHandler(console)

# Custom log format (includes time, level, and thread name)
formatter = LogFormatter(
    fmt="[%(asctime)s] [%(levelname)s] (%(threadName)-10s) %(message)s"
)
console.setFormatter(formatter)


"""Parse the command line arguments"""
parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str)

args = parser.parse_args()

# Read experiment settings
with open(args.config_file, "r") as f:
    experiment_settings = yaml.safe_load(f)

try:
    frequency = float(experiment_settings.get("frequency", 920e6))
    channel = int(experiment_settings.get("channel", 0))
    gain = float(experiment_settings.get("gain", 80))
    rate = float(experiment_settings.get("rate", 250e3))
    duration = int(experiment_settings.get("duration", 10))
except ValueError as e:
    print("Could not read all settings:", e)
    sys.exit(-1)

client = None 
got_sync = False
meas_id = ""

def handle_sync(command, args):
    print("Received SYNC command:", command, args)
    
    global got_sync
    global meas_id
    
    got_sync = True
    meas_id = args[0]
    

def handle_signal(signum, frame):
    print("\nStopping client...")
    client.stop()


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

CLOCK_TIMEOUT = 1000  # 1000mS timeout for external clock locking
REF_RX_CH = FREE_TX_CH = 0
LOOPBACK_RX_CH = LOOPBACK_TX_CH = 1
logger.debug("\nPLL REF → CH0 RX\nCH1 TX → CH1 RX\nCH0 TX →")

def starting_in(usrp, at_time):
    return f"Starting in {delta(usrp, at_time):.2f}s"


def delta(usrp, at_time):
    return at_time - usrp.get_time_now().get_real_secs()


def get_current_time(usrp):
    return usrp.get_time_now().get_real_secs()


def setup_usrp_clock(usrp, clock_src, num_mboards):
    usrp.set_clock_source(clock_src)

    end_time = datetime.now() + timedelta(milliseconds=CLOCK_TIMEOUT)

    print("Now confirming lock on clock signals...")

    # Lock onto clock signals for all mboards
    for i in range(num_mboards):
        is_locked = usrp.get_mboard_sensor("ref_locked", i)
        while (not is_locked) and (datetime.now() < end_time):
            time.sleep(1e-3)
            is_locked = usrp.get_mboard_sensor("ref_locked", i)
        if not is_locked:
            print("Unable to confirm clock signal locked on board %d", i)
            return False
        else:
            print("Clock signals are locked")
    return True


def setup_usrp_pps(usrp, pps):
    """Setup the PPS source"""
    usrp.set_time_source(pps)
    return True


def print_tune_result(tune_res):
    logger.debug(
        "Tune Result:\n    Target RF  Freq: %.6f (MHz)\n Actual RF  Freq: %.6f (MHz)\n Target DSP Freq: %.6f "
        "(MHz)\n "
        "Actual DSP Freq: %.6f (MHz)\n",
        (tune_res.target_rf_freq / 1e6),
        (tune_res.actual_rf_freq / 1e6),
        (tune_res.target_dsp_freq / 1e6),
        (tune_res.actual_dsp_freq / 1e6),
    )


def tune_usrp(usrp, freq, channels, at_time):
    """Synchronously set the device's frequency.
    If a channel is using an internal LO it will be tuned first
    and every other channel will be manually tuned based on the response.
    This is to account for the internal LO channel having an offset in the actual DSP frequency.
    Then all channels are synchronously tuned."""
    treq = uhd.types.TuneRequest(freq)
    usrp.set_command_time(uhd.types.TimeSpec(at_time))
    treq.dsp_freq = 0.0
    treq.target_freq = freq
    treq.rf_freq = freq
    treq.rf_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
    treq.dsp_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
    args = uhd.types.DeviceAddr("mode_n=integer")
    treq.args = args
    rx_freq = freq - 1e3
    rreq = uhd.types.TuneRequest(rx_freq)
    rreq.rf_freq = rx_freq
    rreq.target_freq = rx_freq
    rreq.dsp_freq = 0.0
    rreq.rf_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
    rreq.dsp_freq_policy = uhd.types.TuneRequestPolicy(ord("M"))
    rreq.args = uhd.types.DeviceAddr("mode_n=fractional")
    for chan in channels:
        print_tune_result(usrp.set_rx_freq(rreq, chan))
        print_tune_result(usrp.set_tx_freq(treq, chan))
    while not usrp.get_rx_sensor("lo_locked").to_bool():
        print(".")
        time.sleep(0.01)
    logger.info("RX LO is locked")
    while not usrp.get_tx_sensor("lo_locked").to_bool():
        print(".")
        time.sleep(0.01)
    logger.info("TX LO is locked")


def setup_usrp(usrp, server_ip, connect=True):
    rate = RATE
    mcr = 20e6
    assert (
        mcr / rate
    ).is_integer(), f"The masterclock rate {mcr} should be an integer multiple of the sampling rate {rate}"
    # Manual selection of master clock rate may also be required to synchronize multiple B200 units in time.
    usrp.set_master_clock_rate(mcr)
    channels = [0, 1]
    setup_usrp_clock(usrp, "external", usrp.get_num_mboards())
    setup_usrp_pps(usrp, "external")
    # smallest as possible (https://files.ettus.com/manual/page_usrp_b200.html#b200_fe_bw)
    rx_bw = 200e3
    for chan in channels:
        usrp.set_rx_rate(rate, chan)
        usrp.set_tx_rate(rate, chan)
        # NOTE DC offset is enabled
        usrp.set_rx_dc_offset(True, chan)
        usrp.set_rx_bandwidth(rx_bw, chan)
        usrp.set_rx_agc(False, chan)
    # specific settings from loopback/REF PLL
    usrp.set_tx_gain(LOOPBACK_TX_GAIN, LOOPBACK_TX_CH)
    usrp.set_tx_gain(LOOPBACK_TX_GAIN, FREE_TX_CH)

    usrp.set_rx_gain(LOOPBACK_RX_GAIN, LOOPBACK_RX_CH)
    usrp.set_rx_gain(REF_RX_GAIN, REF_RX_CH)
    # streaming arguments
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = channels
    # streamers
    tx_streamer = usrp.get_tx_stream(st_args)
    rx_streamer = usrp.get_rx_stream(st_args)
    # Step1: wait for the last pps time to transition to catch the edge
    # Step2: set the time at the next pps (synchronous for all boards)
    # this is better than set_time_next_pps as we wait till the next PPS to transition and after that we set the time.
    # this ensures that the FPGA has enough time to clock in the new timespec (otherwise it could be too close to a PPS edge)
    logger.info("Waiting for server sync")
    while not got_sync:
        pass
    
    logger.info("Setting device timestamp to 0...")
    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))

    usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
    logger.debug("[SYNC] Resetting time.")
    logger.info(f"RX GAIN PROFILE CH0: {usrp.get_rx_gain_names(0)}")
    logger.info(f"RX GAIN PROFILE CH1: {usrp.get_rx_gain_names(1)}")
    # we wait 2 seconds to ensure a PPS rising edge occurs and latches the 0.000s value to both USRPs.
    time.sleep(2)
    tune_usrp(usrp, FREQ, channels, at_time=begin_time)
    logger.info(
        f"USRP has been tuned and setup. ({usrp.get_time_now().get_real_secs()})"
    )
    return tx_streamer, rx_streamer


def tx_thread(
    usrp, tx_streamer, quit_event, phase=[0, 0], amplitude=[0.8, 0.8], start_time=None
):
    tx_thr = threading.Thread(
        target=tx_ref,
        args=(usrp, tx_streamer, quit_event, phase, amplitude, start_time),
    )

    tx_thr.name = "TX_thread"
    tx_thr.start()

    return tx_thr


def rx_thread(usrp, rx_streamer, quit_event, duration, res, start_time=None):
    _rx_thread = threading.Thread(
        target=rx_ref,
        args=(
            usrp,
            rx_streamer,
            quit_event,
            duration,
            res,
            start_time,
        ),
    )
    _rx_thread.name = "RX_thread"
    _rx_thread.start()
    return _rx_thread


def measure_loopback(
    usrp, tx_streamer, rx_streamer, quit_event, result_queue, at_time=None
):
    # ------------------------------------------------------------
    # Function: measure_loopback
    # Purpose:
    #   This function performs a loopback measurement using a USRP device.
    #   It transmits a known signal on one channel and simultaneously
    #   receives it on another channel (loopback). The result is captured,
    #   stored, and processed later.
    # ------------------------------------------------------------

    logger.debug("########### Measure LOOPBACK ###########")

    # ------------------------------------------------------------
    # 1. Configure transmit signal amplitudes
    # ------------------------------------------------------------
    amplitudes = [0.0, 0.0]              # Initialize amplitude array for both channels
    amplitudes[LOOPBACK_TX_CH] = 0.8     # Enable TX on the selected loopback channel

    # ------------------------------------------------------------
    # 2. Set the transmission start time
    # ------------------------------------------------------------
    start_time = uhd.types.TimeSpec(at_time)
    logger.debug(starting_in(usrp, at_time))

    # ------------------------------------------------------------
    # 3. (Legacy) Access user settings interface for low-level FPGA control
    #    Used to switch the USRP into "loopback mode" by writing to
    #    a register in the user settings interface.
    #    NOTE: This interface is no longer available in UHD 4.x.
    # ------------------------------------------------------------
    user_settings = None
    try:
        user_settings = usrp.get_user_settings_iface(1)
        if user_settings:
            # Read current register value (for debug)
            logger.debug(user_settings.peek32(0))
            # Write a value to activate loopback mode
            user_settings.poke32(0, SWITCH_LOOPBACK_MODE)
            # Read again to verify the register value was updated
            logger.debug(user_settings.peek32(0))
        else:
            logger.error("Cannot write to user settings.")
    except Exception as e:
        logger.error(e)

    # ------------------------------------------------------------
    # 4. Start transmit (TX), metadata, and receive (RX) threads
    # ------------------------------------------------------------
    tx_thr = tx_thread(
        usrp,
        tx_streamer,
        quit_event,
        amplitude=amplitudes,
        phase=[0.0, 0.0],
        start_time=start_time,
    )

    # Thread responsible for handling TX metadata (timestamps, etc.)
    tx_meta_thr = tx_meta_thread(tx_streamer, quit_event)

    # Thread that captures received samples during loopback
    rx_thr = rx_thread(
        usrp,
        rx_streamer,
        quit_event,
        duration=CAPTURE_TIME,
        res=result_queue,
        start_time=start_time,
    )

    # ------------------------------------------------------------
    # 5. Wait for the capture duration plus some safety margin (delta)
    # ------------------------------------------------------------
    time.sleep(CAPTURE_TIME + delta(usrp, at_time))

    # ------------------------------------------------------------
    # 6. Signal all threads to stop and wait for them to finish
    # ------------------------------------------------------------
    quit_event.set()   # Triggers thread termination
    tx_thr.join()
    rx_thr.join()
    tx_meta_thr.join()

    # ------------------------------------------------------------
    # 7. Reset the RF switch control (disable loopback mode)
    # ------------------------------------------------------------
    if user_settings:
        user_settings.poke32(0, SWITCH_RESET_MODE)

    # ------------------------------------------------------------
    # 8. Clear the quit event flag to prepare for the next measurement
    # ------------------------------------------------------------
    quit_event.clear()


def tx(duration, tx_streamer, rate, channels):
    print("TX START")
    metadata = uhd.types.TXMetadata()

    buffer_samps = tx_streamer.get_max_num_samps()
    samps_to_send = int(rate*duration)

    signal = np.ones((len(channels), buffer_samps), dtype=np.complex64)
    signal *= np.exp(1j*np.random.rand(len(channels), 1)*2*np.pi)*0.8 # 0.8 to not exceed to 1.0 threshold

    print(signal[:,0])

    send_samps = 0

    while send_samps < samps_to_send:
        samples = tx_streamer.send(signal, metadata)
        send_samps += samples
    # Send EOB to terminate Tx
    metadata.end_of_burst = True
    tx_streamer.send(np.zeros((len(channels), 1), dtype=np.complex64), metadata)
    print("TX END")
    # Help the garbage collection
    return send_samps


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        # Attempt to open and load calibration settings from the YAML file
        with open(os.path.join(script_dir, "cal-settings.yml"), "r") as file:
            vars = yaml.safe_load(file)
            globals().update(vars)  # update the global variables with the vars in yaml
    except FileNotFoundError:
        logger.error("Calibration file 'cal-settings.yml' not found in the current directory.")
        exit()
    except yaml.YAMLError as e:
        logger.error(f"Error parsing 'cal-settings.yml': {e}")
        exit()
    except Exception as e:
        logger.error(f"Unexpected error while loading calibration settings: {e}")
        exit()
    
    logger.debug(vars)
    
    try:
        # FPGA file path
        fpga_path = os.path.join(script_dir, "usrp_b210_fpga_loopback.bin")

        # Initialize USRP device with custom FPGA image and integer mode
        usrp = uhd.usrp.MultiUSRP(
            "enable_user_regs, " \
            f"fpga={fpga_path}, " \
            "mode_n=integer"
        )
        logger.info("Using Device: %s", usrp.get_pp_string())
   
        client = Client(args.config_file)
        client.on("SYNC", handle_sync)
        client.start()
        logger.debug("Client running...")
   
        # -------------------------------------------------------------------------
        # STEP 0: Preparations
        # -------------------------------------------------------------------------

        # Set up TX and RX streamers and establish connection
        tx_streamer, rx_streamer = setup_usrp(usrp, server_ip, connect=True)

        print(client.hostname)
        print(meas_id)
        file_name = f"data_{client.hostname}_{meas_id}.txt"

        try:
            data_file = open(file_name, "a")
        except Exception as e:
            logger.error(e)

        # Event used to control thread termination
        quit_event = threading.Event()

        margin = 5.0                     # Safety margin for timing
        cmd_time = CAPTURE_TIME + margin # Duration for one measurement step
        start_next_cmd = cmd_time        # Timestamp for the next scheduled command

        # Queue to collect measurement results and communicate between threads
        result_queue = queue.Queue()
        
        # -------------------------------------------------------------------------
        # STEP 1: Perform loopback measurement with reference signal
        # -------------------------------------------------------------------------

        # # --- Perform pilot measurement ---
        # file_name_state = file_name + "_pilot"
        # measure_pilot(
        #     usrp,
        #     tx_streamer,
        #     rx_streamer,
        #     quit_event,
        #     result_queue,
        #     at_time=start_next_cmd
        # )

        # # Retrieve pilot phase result
        # phi_P = result_queue.get()

        # # Print pilot phase
        # logger.info("Phase pilot reference signal: %s", phi_P)

        start_next_cmd += cmd_time + 1.0  # Schedule next command after delay

        # -------------------------------------------------------------------------
        # STEP 2: Perform internal loopback measurement with reference signal
        # -------------------------------------------------------------------------

        print(uhd.get_version_string())
        
        quit()

        file_name_state = file_name + "_loopback"
        measure_loopback(
            usrp,
            tx_streamer,
            rx_streamer,
            quit_event,
            result_queue,
            at_time=start_next_cmd,
        )

        # Retrieve loopback phase result
        phi_LB = result_queue.get()

        # Print loopback phase
        logger.info("Phase pilot reference signal in rad: %s", phi_LB)
        logger.info("Phase pilot reference signal in degrees: %s", np.rad2deg(phi_LB))

        start_next_cmd += cmd_time + 2.0  # Schedule next command
    
    except Exception as e:
        logger.error(e)
    
    quit()
   


    client = Client(args.config_file)
    client.on("tx-start", handle_tx_start)
    client.start()
    print("Client running...")
    
    try:
        while client.running:
            if got_start:
                got_start = False
                tx(duration, tx_streamer, rate, [channel])
                client.send("tx-done")
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    client.stop()
    client.join()
    print("Client terminated.")
