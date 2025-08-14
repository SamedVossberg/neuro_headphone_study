import pandas as pd
import numpy as np
from mne.filter import filter_data
from mne.channels import read_custom_montage
from mne import create_info
from mne.io import RawArray
from neurokit2 import signal_filter
import importlib.resources

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
    raw = raw.set_montage(MONTAGE)

    if plot_montage:
        raw.plot_sensors(show_names=True)
    return raw