import pandas as pd
import numpy as np
import mne
import neurokit2 as nk
import warnings
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



def _extract_run_number(run_value):
    """Return integer run number from strings like 'rec1' or numerics like 1."""
    s = str(run_value).strip().lower()
    digits = re.sub(r"[^0-7]", "", s)
    return int(digits) if digits else None

def compute_session_avg_psd(active_recording, cond_ec_tag, cond_eo_tag):
    """
    For a single session row from `sessions`, load EEG, label EC/EO segments, compute robust PSD per channel,
    and return two DataFrames (EC and EO) with columns ['Frequency','Power'] averaged across channels.
    """

    eeg, snr_df, load_report = process_recording(active_recording)
    fs = float(load_report.get('estimated_fs', 250.0))

    # Select signals for each condition
    ec_df = eeg[eeg['Condition'].astype(str).str.contains(cond_ec_tag, case=False, regex=False)]
    eo_df = eeg[eeg['Condition'].astype(str).str.contains(cond_eo_tag, case=False, regex=False)]

    if ec_df.empty or eo_df.empty:
        raise ValueError(f"Missing EC/EO segments for tags '{cond_ec_tag}'/'{cond_eo_tag}' in recording {active_recording.ID} ({active_recording.Run}).")

    # Average PSD across channels for each condition -> Same building average of delta or delta after average? -> jup

    desired = chans_ear_only if active_recording.Config == "Ear Only" else chans_top_ear
    use_chans = [c for c in desired if c in eeg.columns]
    if not use_chans:
        raise ValueError(
            f"No desired channels found in EEG for {active_recording.ID} ({active_recording.Run}). "
            f"Desired={desired}, available={[c for c in eeg.columns if c.startswith('A')]}"
        )

    def _avg_psd_over_channels(signals_df):
        psd_list = []
        for ch in use_chans:
            series = signals_df[ch]
            psd_df = flows.compute_robust_psd(series, min_freq=min_freq, max_freq=max_freq, fs=fs,
                                              normalize=True, window_sec=1, overlap_pct=0, verbose='WARNING')
            psd_df = psd_df.rename(columns={'Power': f'P_{ch}'})
            psd_list.append(psd_df)
        # merge on frequency
        merged = psd_list[0]
        for nxt in psd_list[1:]:
            merged = merged.merge(nxt, on='Frequency', how='inner')
        merged['Power'] = merged.filter(like='P_').mean(axis=1)
        return merged[['Frequency','Power']]

    ec_psd = _avg_psd_over_channels(ec_df)
    eo_psd = _avg_psd_over_channels(eo_df)

    return ec_psd, eo_psd, fs

def average_psds_across_sessions(df_sessions, config_label, verbose=True): # TODO: Should we exclude dead or too noisy chans/sessions?
    """
    Compute the average EC and EO PSD across all first sessions for a given configuration.
    Returns a dict with keys: 'EC', 'EO', 'alpha_effect', 'n_used', 'fs_median'.
    The alpha_effect is the integrated alpha-band (8-13 Hz) difference EC minus EO (Berger effect magnitude).
    """
    # Filter sessions
    sel = df_sessions.copy()
    sel = sel[sel['Config'].astype(str) == config_label]

    # First-session filter: accept 'rec1' or numeric 1
    run_mask = sel['Run'].astype(str).str.lower().str.contains("rec1") | (sel['Run'].astype(str).str.replace(r"[^0-7]","", regex=True) == "1")
    sel = sel[run_mask]

    if sel.empty:
        raise ValueError(f"No sessions found for config '{config_label}' and Run == 1.")

    ec_psds = []
    eo_psds = []
    fs_list = []

    for row in sel.itertuples(index=False):
        try:
            run_num = _extract_run_number(getattr(row, "Run"))
            cond_ec = f"headphones_setup_ec_{run_num}" if run_num is not None else "headphones_setup_ec_1" # careful with "ec" or only "eo" -> "rec" === True with "ec"
            cond_eo = f"headphones_setup_eo_{run_num}" if run_num is not None else "headphones_setup_eo_1"
            ec_psd, eo_psd, fs = compute_session_avg_psd(row, cond_ec, cond_eo)
            ec_psds.append(ec_psd)
            eo_psds.append(eo_psd)
            fs_list.append(fs)
            if verbose:
                print(f"✅ Processed {row.ID} ({row.Run}) for '{config_label}'")
        except Exception as e:
            warnings.warn(f"Skipping a session due to error: {e}")

    if len(ec_psds) == 0:
        raise RuntimeError(f"All sessions for config '{config_label}' failed to process.")

    # Establish a common frequency grid (use intersection to be safe)
    common_freqs = set(ec_psds[0]['Frequency'].round(6).values)
    for df in ec_psds[1:] + eo_psds:
        common_freqs &= set(df['Frequency'].round(6).values)
    common_freqs = np.array(sorted(list(common_freqs)))
    if common_freqs.size < 10:
        # Fallback: use first EC grid and interpolate others onto it
        base_freqs = ec_psds[0]['Frequency'].values
        def interp_to_base(df):
            return pd.DataFrame({
                'Frequency': base_freqs,
                'Power': np.interp(base_freqs, df['Frequency'].values, df['Power'].values)
            })
        ec_stack = [interp_to_base(df) for df in ec_psds]
        eo_stack = [interp_to_base(df) for df in eo_psds]
    else:
        def trim_to_common(df):
            mask = np.isin(df['Frequency'].round(6).values, common_freqs)
            return df.loc[mask].sort_values('Frequency').reset_index(drop=True)
        ec_stack = [trim_to_common(df) for df in ec_psds]
        eo_stack = [trim_to_common(df) for df in eo_psds]

    # Average across sessions
    ec_avg = pd.concat(ec_stack).groupby('Frequency', as_index=False)['Power'].mean()
    eo_avg = pd.concat(eo_stack).groupby('Frequency', as_index=False)['Power'].mean()

    # Compute Berger effect magnitude = alpha-band (8–13 Hz) area difference EC - EO
    def band_area(psd_df, lo=8.0, hi=13.0):
        m = (psd_df['Frequency'] >= lo) & (psd_df['Frequency'] <= hi)
        if m.sum() < 2:  # not enough points for integration
            return np.nan
        return float(simpson(psd_df.loc[m, 'Power'].values, x=psd_df.loc[m, 'Frequency'].values))

    alpha_ec = band_area(ec_avg, 8.0, 13.0)
    alpha_eo = band_area(eo_avg, 8.0, 13.0)
    alpha_effect = alpha_ec - alpha_eo  # Berger effect: higher alpha in EC than EO

    return {
        'EC': ec_avg,
        'EO': eo_avg,
        'alpha_effect': alpha_effect,
        'n_used': len(ec_stack),
        'fs_median': float(np.median(fs_list)) if fs_list else np.nan,
    }