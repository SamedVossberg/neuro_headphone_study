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
