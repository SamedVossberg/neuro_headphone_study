import pandas as pd
import numpy as np
import mne
import neurokit2 as nk
from scipy.integrate import simps
from mne.time_frequency import psd_array_welch

def report_chan_amps(eeg_df, report=True, window_width_sec=None, sampling_rate=None):
    # µVrms is also reported in OpenBCI GUI - so used here for comparison
    def compute_rms(df):
        return df.apply(lambda x: np.sqrt(np.mean(x ** 2)), axis=0)

    if window_width_sec is not None:
        if sampling_rate is None:
            raise ValueError("sampling_rate must be provided when using windowed RMS")

        samples_per_window = int(window_width_sec * sampling_rate)
        n_windows = eeg_df.shape[0] // samples_per_window

        # Split into non-overlapping windows
        rms_list = []
        for i in range(n_windows):
            start = i * samples_per_window
            end = start + samples_per_window
            window = eeg_df.iloc[start:end]
            rms_list.append(compute_rms(window))

        # Stack and take median per channel
        rms = pd.DataFrame(rms_list).median().round(2)
    else:
        # Use full data
        rms = compute_rms(eeg_df).round(2)

    if report:
        print('RMS per channel:')
        print(rms)

    # Expressed as quartile coverage (broader than IQR)
    # qr_75 = eeg_df.apply(lambda x: x.quantile(0.875) - x.quantile(0.125), axis=0)
    # print('75% quartile amplitudes per channel:')
    # print(qr_75)

    return rms.to_dict()

def compute_robust_psd(channel_data, min_freq, max_freq, fs, normalize=False, window_sec=2, overlap_pct=0,
                         verbose="WARNING"):
    """
    channel_data = channel_data.interpolate(limit_direction='both') # Fill NaN values using an interpolation method.
    if channel_data.isna().any():
        if verbose:
            print("NaNs remain after interpolation — skipping channel.")
        return None  # Signal unusable
    """

    # Get per-window PSDs
    psds, freqs = psd_array_welch(
        x=channel_data.values,
        sfreq=fs,
        fmin=min_freq,
        fmax=max_freq,
        n_fft=int(window_sec * fs),
        n_overlap=int(window_sec * fs * overlap_pct),
        average="median",  # implemented directly...
        verbose=verbose
    )

    if normalize:
        psds /= psds.sum()

    return pd.DataFrame({'Frequency': freqs, 'Power': psds})

def get_channel_band_power(eeg_signal_series, fs):
    # Ensure the signal has enough length
    if len(eeg_signal_series) < 250:  # Example threshold, can be adjusted
        raise ValueError("Signal length is too short for multitaper PSD estimation.")

    # Compute PSD using multitaper method
    psd, freqs = mne.time_frequency.psd_array_multitaper(np.array(eeg_signal_series), sfreq=fs, fmin=2, fmax=15,
                                                         verbose='WARNING')
    # psd, freqs = mne.time_frequency.psd_array_welch(np.array(eeg_signal_series), sfreq=fs, fmin=min_freq, fmax=max_freq, n_fft=fs*4, n_overlap=int(fs/2), n_per_seg = fs*2, verbose='WARNING')

    # Normalize power spectrum
    # psd = 10 * np.log10(psd)  # convert to dB
    psd /= psd.sum()  # convert to relative power

    # Define frequency bands
    bands = {
        'Theta': (4, 7),
        'Alpha': (8, 13)
    }

    # Could use IAF-based freq band ranges by providing them as a param here and then use
    # freq_bands = {'theta':[peak_theta-1,peak_theta+1],
    #               'alpha':[peak_alpha-1.5,peak_alpha+1.5]}

    # Calculate band powers using trapezoid integral estimation
    # band_powers = {band: np.trapz(psd[(freqs >= low) & (freqs <= high)], freqs[(freqs >= low) & (freqs <= high)])
    # for band, (low, high) in bands.items()}
    # Calculate band powers using Simpson's rule
    band_powers = {band: simps(psd[(freqs >= low) & (freqs <= high)], freqs[(freqs >= low) & (freqs <= high)])
                   for band, (low, high) in bands.items()}

    # Convert to DataFrame
    band_powers_df = pd.DataFrame([band_powers])
    return band_powers_df

def assr_snr(signal, min_freq, max_freq, fs, normalize, window_sec, verbose="WARNING"):
    '''
        Mikkelsen2015: Using the ASSR paradigm, we then estimated the signal-to-
        noise ratios (SNR) for both scalp and ear-EEG setups, whereby
        the SNR was defined as the diﬀerence between the logarithm of
        the power at 40 Hz (the signal) and the logarithm of the average
        power in 5 Hz intervals around 40 Hz (the noise floor).
    '''

    # Make sure we're working on a copy - to avoid warnings
    signal = signal.copy()

    # Calculate PSD
    psd = nk.signal_psd(signal, sampling_rate=fs, min_frequency=min_freq, max_frequency=max_freq, normalize=normalize, window=window_sec)

    # Power values in µV²/Hz → convert to dB
    # TODO: Would have to do that before though (above when the psd is extracted)
    # psd['Power'] = 10 * np.log10(psd['Power'])

    noise_below = psd.query("36 <= Frequency <= 40.5").Power.mean()
    signal_power = psd.query("40.5 < Frequency < 41.5").Power.mean()
    noise_above = psd.query("41.5 <= Frequency <= 46").Power.mean()

    snr = signal_power - np.mean([noise_below, noise_above])
    return snr

def alpha_prominence_snr(signal, min_freq, max_freq, fs, normalize, window_sec, overlap_pct,
                           alpha_band=(8, 12), lower_band=(5, 7), upper_band=(14, 16), verbose="WARNING"):
    # TODO: Could adjust to also use the compute_robust_psd() function...
    # Get per-window PSDs
    psd, freqs = psd_array_welch(
        x=signal.values,
        sfreq=fs,
        fmin=min_freq,
        fmax=max_freq,
        n_fft=int(window_sec * fs),
        n_overlap=int(window_sec * fs * overlap_pct),
        average="median",
        verbose=verbose
    )

    if normalize:
        psd /= psd.sum()

    def band_power(f, p, band):
        mask = (f >= band[0]) & (f <= band[1])
        return np.mean(p[mask])  # mean is more robust than sum for power

    alpha = band_power(freqs, psd, alpha_band)
    flank_low = band_power(freqs, psd, lower_band)
    flank_high = band_power(freqs, psd, upper_band)
    flank_mean = np.mean([flank_low, flank_high])

    return alpha / flank_mean if flank_mean > 0 else np.nan

def extract_IAF(raw, show_psd=None):
    # https://pubmed.ncbi.nlm.nih.gov/29357113/  → Uses the SGF filter for peak detection
    # The transfreq package is rather for getting the transition frequency, so doesn’t really return peak alpha…

    # Compute power spectrum
    # Testing the SGF detection using the cocorana/philisine mne package
    from philistine.mne import savgol_iaf
    iaf = savgol_iaf(raw, picks=None, fmin=6, fmax=14, ax=show_psd)  # ax specifies whether a plot should show up
    return iaf.CenterOfGravity

def scan_assr_windows(x, *, fs=250.0, win_sec=5.0, step_sec=1.0, **snr_kwargs):
    """Slide a window over the signal and compute SNR(dB) per window.
    Returns a DataFrame with columns: t_start, t_end, snr_db
    """
    x = np.asarray(x)
    w = int(round(win_sec * fs))
    s = int(round(step_sec * fs))
    rows = []
    for start in range(0, max(1, len(x) - w + 1), s):
        seg = x[start:start + w]
        snr_db = assr_snr_db(seg, fs=fs, **snr_kwargs)
        rows.append({"t_start": start / fs, "t_end": (start + w) / fs, "snr_db": snr_db})
    return pd.DataFrame(rows)

def assr_snr_db(segment, *, fs=250.0, f0=40.0, half_width=0.5, flank=4.0,
                min_f=1.0, max_f=100.0):
    """Compute ASSR SNR (dB) for a single window around f0.
    signal band = [f0 - half_width, f0 + half_width]
    noise  band = union of [f0 - flank, f0 - half_width) and (f0 + half_width, f0 + flank]
    """
    seg = np.asarray(segment)
    if seg.size < 2:
        return np.nan

    psd = nk.signal_psd(seg, sampling_rate=fs, min_frequency=min_f,max_frequency=max_f, normalize=False, window=1)

    sig = psd.loc[psd["Frequency"].between(f0 - half_width, f0 + half_width, inclusive="both"), "Power"].mean()
    lo  = psd.loc[psd["Frequency"].between(f0 - flank,     f0 - half_width, inclusive="left"), "Power"].mean()
    hi  = psd.loc[psd["Frequency"].between(f0 + half_width, f0 + flank,     inclusive="right"), "Power"].mean()
    noise = np.nanmean([lo, hi])
    if not np.isfinite(sig) or not np.isfinite(noise) or noise <= 0:
        return np.nan
    return float(10 * np.log10(sig / noise))

def to_db(psd_df, power_col='Power'):
    """Return a copy where 'Power' is converted to dB (10*log10)."""
    df = psd_df.copy()
    # Protect against log of zero
    p = np.maximum(df[power_col].to_numpy(dtype=float), 1e-12)
    df['Power'] = 10.0 * np.log10(p)
    return df[['Frequency', 'Power']]

def band_area(psd_df, lo, hi, power_col='Power'):
    """Integrate power in [lo, hi] using trapezoid rule."""
    f = psd_df['Frequency'].to_numpy(dtype=float)
    P = psd_df[power_col].to_numpy(dtype=float)
    m = (f >= lo) & (f <= hi)
    if m.sum() < 2:
        return np.nan
    return float(np.trapz(P[m], f[m]))

def rel_band_power(psd_df, lo, hi, power_col='Power'):
    """Relative band power = area(lo..hi) / area(total)."""
    f = psd_df['Frequency'].to_numpy(dtype=float)
    P = psd_df[power_col].to_numpy(dtype=float)
    if len(f) < 2:
        return np.nan
    band = band_area(psd_df, lo, hi, power_col=power_col)
    total = float(np.trapz(P, f))
    return (band / total) if total > 0 else np.nan