import pandas as pd
import numpy as np
import mne
import re
from mne.filter import filter_data, notch_filter
from mne.channels import read_custom_montage
from mne import create_info
from mne.io import RawArray
from pyprep.find_noisy_channels import NoisyChannels
from asrpy import ASR
from neurokit2 import signal_filter
import importlib.resources
import flows
import warnings

# Pre-process the data
def initial_detrending(eeg, fs, low_freq = 1, method="mne"):
    # Mean center the data
    eeg = eeg.apply(lambda x: x - x.mean())

    # High-pass
    if method == "mne":
        eeg = eeg.apply(
            lambda x: filter_data(np.array(x), sfreq=fs, l_freq=low_freq, h_freq=None, method='fir', copy=True,
                                  verbose='WARNING'))
    elif method == "nk":
        eeg = eeg.apply(
            lambda x:signal_filter(np.array(x), sampling_rate=fs, lowcut=low_freq, highcut=None, method="fir")
        )
    else:
        raise Exception("method must be 'mne' or 'nk'")

    return eeg

# Removing oscillatory components (line noise or other)
# TODO: Not tested yet - but included because it already shows various options
# - Beware: Oscillatory components do not produce lower-frequency harmonics (e.g. 50Hz noise only creates 100Hz harmonics, not 25Hz)
def remove_oscillatory_noise(eeg, fs, eeg_cols):
    # print('Line noise filter now!')
    # Detect 31.25 Hz noise (from impedance check)
    # eeg.loc[:, eeg_cols].apply(lambda x : notch_filter(np.array(x), Fs=fs, freqs=None, method='spectrum_fit'))
    # Could find a method for logging this -> ChatGPT

    # Simple FIR Notch filter
    # eeg.loc[:, eeg_cols] = eeg.loc[:, eeg_cols].apply(lambda x : notch_filter(np.array(x), Fs=fs, freqs=[15.625, 25, 31.25, 50], method='fir', copy=True, verbose='WARNING'))

    # CleanLine filterinig
    # eeg.loc[:, eeg_cols] = eeg.loc[:, eeg_cols].apply(lambda x : notch_filter(np.array(x), Fs=fs, freqs=[15.625, 31.25], method='spectrum_fit', copy=True, verbose='WARNING'))

    # ZapLine
    # Supposedly very good for line noise removal!
    # https://www.sciencedirect.com/science/article/pii/S1053811919309474
    # https://nbara.github.io/python-meegkit/auto_examples/example_dss_line.html#sphx-glr-auto-examples-example-dss-line-py
    # print(eeg.loc[:, eeg_cols].isna().any().any())
    eeg_denoised, _ = dss.dss_line(np.array(eeg.filter(eeg_cols)), 50, sfreq=fs, nremove=2)
    eeg_denoised, _ = dss.dss_line(eeg_denoised, 31.25, sfreq=fs, nremove=1)
    # eeg_denoised, _ = dss.dss_line_iter(np.array(eeg.filter(eeg_cols)), 50, sfreq=fs)
    eeg_denoised = pd.DataFrame(eeg_denoised, columns=eeg_cols)
    eeg_denoised.index = eeg.index  # Give the same index to enable next assignment is correct
    eeg.loc[:, eeg_cols] = eeg_denoised

    return eeg

# Set the reference for headphone EEG (two classic options: Linked Mastoid or Common Average Reference - CAR)
# MONTAGE = read_custom_montage('headphonesPos_sph_all.txt')
# This assumes your text file is in the same package as this code (e.g., flows/)
with importlib.resources.path('flows', 'headphonesPos_sph_all.txt') as file_path:
    MONTAGE = read_custom_montage(str(file_path))

def set_headphones_reference(raw, type):
    if type == "car":
        # Apply the Common Average Reference (CAR)
        raw.set_eeg_reference('average')
        return raw
    elif type == "mastoid":
        # Apply the linked mastoid reference (L4+R4) that is possible in all current headphone default configs
        # Retrieving the original REF channel
        # raw = mne.add_reference_channels(raw, ref_channels=[orig_ref])
        new_ref = ['L4', 'R4']
        raw.set_eeg_reference(ref_channels=new_ref)
        raw = raw.drop_channels(['L4', 'R4'])  # Drop the ref elecs
        return raw
    else:
        raise Exception("Type must be 'car' or 'mastoid'")

def convert_df_to_raw(eeg, fs, eeg_cols, plot_montage=False):
    # Convert to mne object for some more transformation
    ch_names = eeg.filter(eeg_cols).columns.values.tolist()
    info = create_info(ch_names, fs, ch_types='eeg')
    raw = RawArray(np.array(eeg.filter(eeg_cols)).transpose(), info, verbose='WARNING')
    raw = raw.set_montage(MONTAGE, on_missing='ignore')

    if plot_montage:
        raw.plot_sensors(show_names=True)
    return raw

def preprocess_eeg_mne(
    eeg_df: pd.DataFrame,
    sfreq: float,
    ch_names=None,                      # list of channel names or None -> auto-detect numeric EEG columns otherwise problems with TS_Unix col
    time_cols=("TS_UNIX", "TS_Unix", "timestamp", "time", "TIME"),
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch: float | None = 50.0,
    average_ref: bool = True,
    auto_bad: bool = True,
    return_raw: bool = False,   
):
    """
    Minimal MNE pipeline:
      - picks only EEG channels (excludes time/id columns)
      - converts µV->V if needed (heuristic), then filters in V
      - average reference (optional)
      - notch at 50 Hz (optional)
      - band-pass FIR [l_freq, h_freq]
      - light variance-based bad-channel interpolation (optional)
    Returns:
      eeg_df_out (DataFrame with same index & preserved time columns)
      report (dict)
      [raw] (optional if return_raw=True)
    """
    if not isinstance(eeg_df, pd.DataFrame):
        raise TypeError("eeg_df must be a pandas DataFrame")

    # Identify time/meta columns to keep intact
    time_cols_set = {c for c in time_cols if c in eeg_df.columns}
    # Auto-detect EEG channels if not provided: numeric columns minus time/meta columns
    if ch_names is None:
        numeric_cols = eeg_df.select_dtypes(include=[np.number]).columns.tolist()
        ch_names = [c for c in numeric_cols if c not in time_cols_set]
    else:
        # Ensure we don't pass time/meta columns even if they appear in ch_names
        ch_names = [c for c in ch_names if c in eeg_df.columns and c not in time_cols_set]

    if not ch_names:
        raise ValueError("No EEG channels found after excluding time/meta columns.")

    # Build data matrix (n_ch, n_times)
    X_in = eeg_df[ch_names].to_numpy(dtype=float).T

    # Heuristic unit check: if values look like µV, convert to V for filtering
    std_est = float(np.nanstd(X_in))
    convert_to_volts = std_est > 3e-5 
    scale_in = 1e-6 
    X_v = X_in * 1e-6 

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(X_v, info, verbose=False)

    try:
        raw.set_montage("standard_1020", match_case=False, on_missing="ignore")
    except Exception:
        pass

    if average_ref:
        raw.set_eeg_reference("average", verbose=False)

    if notch:
        raw.notch_filter(freqs=[notch], picks="eeg", verbose=False, filter_length = "auto")

    raw.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", method="fir", phase="zero", verbose=False)

    # Light bad-channel handling: robust variance outliers / (nearly) flat channels
    bads = []
    if auto_bad:
        variances = raw.get_data().var(axis=1)
        med = np.median(variances)
        mad = np.median(np.abs(variances - med)) or 1.0
        z_var = 0.67448975 * (variances - med) / mad
        for ch_name, z, v in zip(ch_names, z_var, variances):
            if z > 6.0 or z < -6.0 or v < 1e-12:
                bads.append(ch_name)

        raw.info["bads"] = bads
        if bads:
            raw.interpolate_bads(reset_bads=False, verbose=False)

    # Bring filtered data back to the original unit scale for the DataFrame
    X_v_filt = raw.get_data()                         # V
    X_out = X_v_filt / scale_in                       # back to input units

    eeg_df_out = eeg_df.copy()
    eeg_df_out.loc[:, ch_names] = X_out.T             # preserve index alignment

    report = {
        "sfreq": sfreq,
        "n_channels": len(ch_names),
        "channels": ch_names,
        "bad_channels": bads,
        "input_unit": "µV" if convert_to_volts else "V",
        "returned_unit": "µV" if convert_to_volts else "V",
        "l_freq": l_freq,
        "h_freq": h_freq,
        "notch": notch,
        "average_ref": average_ref,
    }

    if return_raw:
        return eeg_df_out, report, raw
    return eeg_df_out, report





def preprocess_eeg(eeg_df, sfreq: float, return_raw=False, chans=None, notch: float | None = 50.0, l_freq=2, h_freq=50,  average_ref: bool = True):    
    print('--- Pre-processing EEG')
    
    eeg_df = eeg_df.copy() # Make copy to silence warnings
    
    # Mean center the data
    eeg_df = eeg_df.apply(lambda x: x - x.mean()) 

    # Notch-Filter
    # Not needed if focus on <50Hz data anyway
    try:
        if notch:
            eeg_df = eeg_df.apply(lambda x: notch_filter(np.array(x), Fs=sfreq, freqs=[50], copy=True, verbose='WARNING'))
    except Exception as e: 
        print(f'No notch filtering applied {e}')
    
    # Band-Pass Filter the data
    try:
        eeg_df = eeg_df.apply(lambda x: filter_data(np.array(x), sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, copy=True, verbose='WARNING'))
    except Exception as e: 
        print(f'Band-Pass filter not applied due to {e}')
    # Convert to mne object for some more transformation
    raw = flows.convert_df_to_raw(eeg_df, sfreq, chans)
    
    # Detect bad channels using the pyPrep methods for bad channel detection
    try:
        # First stage - remove and interpolate clear outlier channels
        bad_amp_chs = flows.get_rms_bads(eeg_df[chans], sfreq, window_width_sec=1, rms_thresh = 50, report=False)
        print("Clearly bad: " + str(bad_amp_chs)) # Report identified bad chans
        if bad_amp_chs:
            raw.info["bads"] = bad_amp_chs
            raw.interpolate_bads(reset_bads=True)  # interpolate & clear "bads" list
        
        # Second stage - a more refined version of bad channel detection
        nd = NoisyChannels(raw, random_state=42, do_detrend=False) # Assumes that high-pass filter was added before
        nd.find_bad_by_nan_flat()
        nd.find_bad_by_deviation(deviation_threshold=5.0) # 5 is the default
        # nd.find_bad_by_SNR()
        # nd.find_bad_by_hfnoise()
        nd.find_bad_by_correlation(correlation_secs=1.0, correlation_threshold=0.3, frac_bad=0.05) # Default vals
        raw.info['bads'] = nd.get_bads()
        print(raw.info['bads']) # Report identified bad chans

        # Interpolate bad channels
        raw.interpolate_bads(verbose='ERROR')
    except Exception as e:
        print(f"Bad channel detection failed: {e}")
    
    # Re-reference the data to remove monolateral REF
    # Doing it after bad chan detection interpolations ensures robust referencing
    # Removed for now due to odd Berger effect observations...
    raw = flows.set_headphones_reference(raw, type='car') # 'mastoid' or 'average'
        
    # Apply ASR for cleaning sporadic artefacts
    # https://digyt.github.io/asrpy/asrpy/asr.html#asrpy.asr.ASR
    try:
        asr = ASR(sfreq=sfreq, cutoff=10, max_bad_chans=0.3)
        asr.fit(raw)
        raw = asr.transform(raw)
    except Exception as e:
        print(f"ASR failed: {e}")
    
    if return_raw:
        return raw
    else:
        return raw.to_data_frame().drop(['time'], axis=1)