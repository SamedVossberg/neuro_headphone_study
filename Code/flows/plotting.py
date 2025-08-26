import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import neurokit2 as nk
from .analyzing import compute_robust_psd

CHANS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]

# Define a consistent color map for the columns (signals/channels)
colors = plt.cm.get_cmap('tab10', len(CHANS)).colors  # Generate unique colors for each channel
# Convert RGBA to HEX and create the color map
color_map = {chan: f'#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}' for chan, c in
             zip(CHANS, colors)}

# Visualize Signals
def plot_eeg_signals(signals_df):
    # Make sure we're working on a copy - to avoid warnings
    signals_df = signals_df.copy()

    # Melt the DataFrame to long format
    signals_df['index'] = signals_df.index

    signals_df_long = signals_df.melt(id_vars='index', var_name='Signal', value_name='Value')

    # Create the line plot
    fig = px.line(signals_df_long, x='index', y='Value', color='Signal', facet_row='Signal',
                  labels={'time': 'Time', 'Value': 'Signal Value'},
                  color_discrete_map=color_map, render_mode='webgl')

    # Update layout for better spacing
    fig.update_layout(title='EEG Signals', showlegend=False)
    # fig.update_yaxes(matches=None)

    # Show the plot
    fig.show()

# Helper function to generate 1/f curve
def generate_1f_curve_df(psd_df, exponent=2):
    """
    Generate a simple global 1/f reference curve based on an existing PSD DataFrame.

    Parameters:
        psd_df (DataFrame): Must contain 'Frequency' and 'Power' columns.
        exponent (float): The exponent used in the 1/f curve (default = 2.0 for resting state data).

    Returns:
        DataFrame with columns ['Frequency', 'Power'] representing the 1/f line.
    """

    # Determine frequency range and resolution
    freqs = psd_df['Frequency'].values
    min_freq = freqs.min()
    max_freq = freqs.max()

    # Estimate number of points based on resolution
    unique_freqs = np.unique(freqs)
    if len(unique_freqs) < 2:
        raise ValueError("PSD data does not contain enough unique frequency values.")
    resolution = np.median(np.diff(unique_freqs))
    n_points = int((max_freq - min_freq) / resolution)

    # Generate 1/f curve
    f_range = np.linspace(min_freq, max_freq, n_points)
    power = 1 / (f_range ** exponent)
    power /= power.max()
    power *= psd_df['Power'].max()  # Match scale of your PSD data

    return pd.DataFrame({'Frequency': f_range, 'Power': power})

# Visualize PSDs
def plot_eeg_psds(signals_df, min_freq, max_freq, normalize, fs, verbose=True):
    # Make sure we're working on a copy - to avoid warnings
    signals_df = signals_df.copy()
    # print(signals_df.head())

    # Function to compute PSD and format the result
    def compute_psd(column):
        if column.isna().sum() > 0:
            column = column.interpolate()
            if verbose: print("Found NaNs in column - interpolating them")
        # print(column.isna().sum())
        # print(column.dtype)
        return nk.signal_psd(column, sampling_rate=fs,
                             min_frequency=min_freq, max_frequency=max_freq, normalize=normalize)

    # Apply the function to each EEG column
    results = [compute_psd(signals_df[col]).assign(Column=col) for col in signals_df.columns]
    combined_df = pd.concat(results, axis=0).reset_index(drop=True)

    # Pivot the DataFrame to have a single "Frequency" column and multiple "Power" columns
    final_df = combined_df.pivot(index='Frequency', columns='Column', values='Power').reset_index()

    # Now visualize
    fig = px.line(final_df.melt(id_vars='Frequency', var_name='Channel', value_name='Power'),
                  x='Frequency', y='Power', color='Channel', color_discrete_map=color_map)
    fig.show()

# This method uses window median aggrregation to give visualization that is robust agains outliers
def plot_eeg_psds_robust(signals_df, min_freq, max_freq, fs, normalize=False, window_sec=2, overlap_pct=0,
                         verbose="WARNING"):
    """
    Plots robust EEG power spectral densities using MNE with median aggregation.

    Parameters:
        signals_df (DataFrame): Channels as columns, samples as rows.
        min_freq (float): Minimum frequency to include in PSD.
        max_freq (float): Maximum frequency to include in PSD.
        fs (int): Sampling rate in Hz.
        normalize (bool): Whether to normalize power (log scale recommended instead).
        window_sec (float): Window length in seconds for Welch segments.
        verbose (bool): Print interpolation info if NaNs are found.
    """
    # Make sure we're working on a copy - to avoid warnings
    signals_df = signals_df.copy()

    # Compute median PSD for each channel
    all_psds = [
        compute_robust_psd(signals_df[col], min_freq, max_freq, fs, normalize, window_sec, overlap_pct, verbose).assign(Channel=col)
        for col in signals_df.columns
    ]
    combined_df = pd.concat(all_psds, ignore_index=True)

    # Plot
    fig = px.line(
        combined_df,
        x='Frequency', y='Power', color='Channel',
        title='Median-Aggregated EEG Power Spectral Density'
    )

    # Add 1/f distribution curve
    one_over_f_df = generate_1f_curve_df(combined_df)

    fig.add_scatter(
        x=one_over_f_df['Frequency'],
        y=one_over_f_df['Power'],
        mode='lines',
        name='1/f reference',
        line=dict(dash='dash', color='black')
    )

    fig.show()

def plot_PSD_two_conditions(eeg_df, chans, condition1, condition2,
                            fs, min_freq, max_freq, normalize):
    # Make sure we're working on a copy - to avoid warnings
    eeg_df = eeg_df.copy()

    # Filter the data for the conditions of interest
    eeg_cond1 = eeg_df[eeg_df.Condition.str.contains(condition1)]
    eeg_cond2 = eeg_df[eeg_df.Condition.str.contains(condition2)]

    # Function to compute PSD and format the result
    def extract_avg_psd(signals_df, condition_name):
        # Apply the function to each EEG column
        results = [nk.signal_psd(signals_df[col],
                                 sampling_rate=fs, min_frequency=min_freq,
                                 max_frequency=max_freq, normalize=normalize).assign(Column=col) for col in
                   signals_df.columns]

        combined_df = pd.concat(results, axis=0).reset_index(drop=True)

        # Pivot the DataFrame to have a single "Frequency" column and multiple "Power" columns
        # combined_df = combined_df.pivot(index='Frequency', columns='Column', values='Power').reset_index() # This was before averaging the signals
        combined_df = combined_df.pivot(index='Frequency', columns='Column', values='Power').mean(axis=1).reset_index()
        combined_df = combined_df.rename(columns={0: condition_name})
        return combined_df

    psd_df = pd.merge(extract_avg_psd(eeg_cond1[chans], condition1), extract_avg_psd(eeg_cond2[chans], condition2))
    # display(psd_df)

    # Now visualize
    fig = px.line(psd_df.melt(id_vars='Frequency', var_name='Condition', value_name='Power'),
                  x='Frequency', y='Power', color='Condition', color_discrete_map=color_map)
    fig.show()


def plot_assr_snr_over_time(snr_df, *, snr_db_thresh=3.0, title="ASSR SNR over time"):
    fig = px.line(snr_df, x="t_start", y="snr_db", markers=True, title=f"{title} (threshold = {snr_db_thresh} dB)")
    fig.add_hline(y=snr_db_thresh, line_dash="dot")
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="SNR (dB)")
    return fig

def plot_psd_bar_for_window(x, *, fs=250.0, t_start=0.0, win_sec=5.0,
                            max_freq=100.0, bin_width=1.0, f0=40.0,
                            min_frequency=7.5):
    # Extract window
    x = np.asarray(x)
    seg = x[int(t_start*fs): int((t_start+win_sec)*fs)]
    if seg.size < 2:
        raise ValueError("Selected window is too short.")
    # Compute PSD and bin
    psd = nk.signal_psd(seg, sampling_rate=fs, min_frequency=min_frequency, max_frequency=float(max_freq),
                        normalize=False, window=1)
    psd = psd[(psd["Frequency"] >= min_frequency) & (psd["Frequency"] <= max_freq)].copy()

    edges = np.arange(min_frequency, max_freq + bin_width, bin_width)
    psd["bin"] = pd.cut(psd["Frequency"], bins=edges, right=False, include_lowest=True)
    binned = psd.groupby("bin", as_index=False)["Power"].mean()
    binned["Hz_mid"] = binned["bin"].apply(lambda iv: (iv.left + iv.right) / 2)

    title = f"PSD (bar) for window {t_start:.1f}-{t_start+win_sec:.1f}s"
    fig = px.bar(binned, x="Hz_mid", y="Power",
                 labels={"Hz_mid": "Frequency (Hz)", "Power": "Power"}, title=title)
    fig.add_vline(x=float(f0), line_dash="dot")
    fig.update_layout(xaxis=dict(range=[min_frequency, max_freq]))
    return fig

def plot_avg_psds_by_config_px(ear_only, top_ears, *,
                               x_range=(0, 30), y_label="Power",
                               title="Average PSDs: EC vs EO per configuration",
                               show=True):
    # Build a tidy DataFrame with 4 lines: EarOnly-EC/EO, Top+Ears-EC/EO
    df = pd.concat([
        pd.DataFrame({"Frequency": ear_only["EC"]["Frequency"], "Power": ear_only["EC"]["Power"], "State": "EC", "Config": "Ear Only"}),
        pd.DataFrame({"Frequency": ear_only["EO"]["Frequency"], "Power": ear_only["EO"]["Power"], "State": "EO", "Config": "Ear Only"}),
        pd.DataFrame({"Frequency": top_ears["EC"]["Frequency"], "Power": top_ears["EC"]["Power"], "State": "EC", "Config": "Top+Ears"}),
        pd.DataFrame({"Frequency": top_ears["EO"]["Frequency"], "Power": top_ears["EO"]["Power"], "State": "EO", "Config": "Top+Ears"}),
    ], ignore_index=True)

    fig = px.line(df, x="Frequency", y="Power", color="Config", line_dash="State",
                  labels={"Frequency": "Frequency (Hz)", "Power": y_label},
                  title=title)
    fig.update_xaxes(range=list(x_range))
    fig.update_yaxes(showgrid=True)
    if show: 
        fig.show()
        return None
    return fig


def plot_effect_bars_px(values_by_config: dict, *,
                        y_label: str, title: str, text_fmt="{:.3f}", show=True):
    # values_by_config e.g. {"Ear Only": 0.12, "Top+Ears": 0.08}
    df = pd.DataFrame({"Config": list(values_by_config.keys()),
                       "Value": list(values_by_config.values())})
    df["Text"] = [text_fmt.format(v) for v in df["Value"]]
    fig = px.bar(df, x="Config", y="Value", text="Text",
                 labels={"Value": y_label}, title=title)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(showgrid=True)
    if show: 
        fig.show()
        return None
    return fig