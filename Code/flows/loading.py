import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path

EXTRA_COLS = 'Index|Datetime|timestamp|sampleNumber|Accel|Note'

# Streams SETUP Stage ----------
def load_session_setup_data(folder_path, chanlist=None, cleanFile=True):
    # Find all files in the folder that contain "Setup" in the name
    folder = Path(folder_path)
    setup_files = list(folder.glob("*Setup*.csv"))

    # Load and format the data
    eeg = pd.read_csv(setup_files[0], sep=";", low_memory=False)

    # Reformat timestamps
    eeg['TS_UNIX'] = pd.to_datetime(eeg['timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Berlin')

    if cleanFile:
        # Drop some cols
        eeg = eeg[eeg.columns.drop(list(eeg.filter(regex=EXTRA_COLS)))]

        # Adjust channel names
        if chanlist:
            eeg.columns = ['TS_UNIX'] + chanlist
        else:
            eeg.columns = ['TS_UNIX'] + ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]

    return eeg

def extract_assr_segment(df, cleanFile=True):
    i = df.index[df["Note"] == "Audio Recording Started"][0]
    j = df.index[df["Note"] == "Audio Recording Ended"][0]
    df = df.iloc[i + 1:j]

    if cleanFile:
        df = df[df.columns.drop(list(df.filter(regex=EXTRA_COLS)))]  # Drop some cols

    return df

# Streams RECORDING Stage ----------
# Using the version form headphones V2 2025 (from 05/2025)
def load_exg_streams_data(folder_path, chanlist=None, report=True, cleanFile=True):
    # Find all files in the folder that contain "Setup" in the name
    folder = Path(folder_path)
    exg_files = list(folder.glob("*Recording*.csv"))

    eeg = pd.read_csv(exg_files[0], sep=";", low_memory=False)

    # Report sample loss
    sample_count_diffs = eeg.sampleNumber.diff()
    sample_count_diffs = sample_count_diffs[
                             (sample_count_diffs != -255) & (sample_count_diffs != 1)] - 1  # Filter expected
    n_times_lost = sample_count_diffs.count()
    total_samples_lost = sample_count_diffs.sum()

    if report:
        print(f"n times samples loss: {n_times_lost}")
        print(f"Total samples lost: {total_samples_lost}")

    # Reformat timestamps
    eeg['TS_UNIX'] = pd.to_datetime(eeg['timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Berlin')

    if cleanFile:
        # Drop some cols
        eeg = eeg[eeg.columns.drop(list(eeg.filter(regex=EXTRA_COLS)))]

        # Adjust channel names
        if chanlist:
            eeg.columns = chanlist + ['TS_UNIX']
        else:
            eeg.columns = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"] + ['TS_UNIX']

    # Show markers
    # distinct_notes = eeg['Note'].dropna().nunique()
    # print(distinct_notes)
    # display(eeg[eeg['Note'].notna()])

    # Estimated fs
    duration_secs = (eeg.TS_UNIX.iloc[-1] - eeg.TS_UNIX.iloc[0]).total_seconds()
    estimated_fs = eeg.shape[0] / duration_secs
    if report:
        print('Recording duration (hh:mm:ss.ms): ' + str(timedelta(seconds=duration_secs)))
        print('Estimated fs: ' + str(round(estimated_fs, 2)))

    # Data gaps
    time_diffs = eeg['TS_UNIX'].diff()
    gaps = time_diffs > pd.Timedelta(seconds=1)  # Identify where the differences are greater than 1 second
    gap_count = gaps.sum()  # Count the number of gaps longer than 1 second
    if report:
        print('n times gaps with >1sec: ' + str(gap_count))

    # Chunky timestamp structure warning
    ts_diffs = np.diff(eeg['TS_UNIX'])
    if report:
        if (time_diffs == 0).mean() > 0.9:
            print("⚠️ Warning: >90% of timestamps are repeated (likely chunked data).")

    # Prep report dictionary
    report_dict = {'duration_secs': duration_secs,
                   'n_times_lost': n_times_lost,
                   'total_samples_lost': total_samples_lost,
                   'estimated_fs': estimated_fs,
                   'gap_count': gap_count}

    return eeg, report_dict

# Streams IMPEDANCES Stage ----------
def load_session_impedances(folder_path):
    # Find all files in the folder that contain "impedance" in the name
    folder = Path(folder_path)
    impedance_files = list(folder.glob("*Impedance*.csv"))

    # Load and concatenate the files
    dfs = [pd.read_csv(file, sep=';', index_col=False) for file in impedance_files]
    combined_df = pd.concat(dfs, ignore_index=True).drop(['Index'], axis=1)

    # Round all numeric columns to 0 decimal places
    numeric_cols = combined_df.select_dtypes(include='number').columns
    combined_df[numeric_cols] = combined_df[numeric_cols].round(0).astype('Int64')

    # Sort by time
    combined_df['Datetime'] = pd.to_datetime(combined_df['Datetime'], utc=True).dt.tz_convert('Europe/Berlin')
    combined_df = combined_df.sort_values(by='Datetime', ascending=True).reset_index(drop=True)

    return combined_df

# CLASSIC OPENBCI GUI Files ----------
def load_obci_gui_data(exg_file_path, chanlist=None, report=True, cleanFile=True):
    # Load the file
    eeg = pd.read_csv(exg_file_path, skiprows=4, low_memory=False)

    # Check sample completeness
    if report:
        # Report sample loss
        sample_count_diffs = eeg.filter(like='Sample', axis=1).diff()
        sample_count_diffs = sample_count_diffs[
                                 (sample_count_diffs != -255) & (sample_count_diffs != 1)] - 1  # Filter expected
        n_times_lost = sample_count_diffs.count()
        total_samples_lost = sample_count_diffs.sum()

        print(f"n times samples loss: {n_times_lost.item()}")
        print(f"Total samples lost: {total_samples_lost.item()}")

    if cleanFile:
        # Drop some cols
        eeg = eeg[eeg.columns.drop(list(eeg.filter(regex='Accel|Other|Analog|Sample|Format')))]

        # Adjust header descriptions
        if chanlist:
            eeg.columns =  chanlist + ['TS_UNIX']
        else:
            eeg.columns = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"] + ['TS_UNIX']

        # Reformat timestamps
        # See: https://www.eldvyn.com/2020/08/converting-number-from-scientific-e.html
        eeg['TS_UNIX'] = eeg['TS_UNIX'] * 1000
        eeg['TS_UNIX'] = pd.to_datetime(eeg['TS_UNIX'], unit='ms', utc=True).dt.tz_convert('Europe/Berlin')

    # Inspect Timestamp information
    if report:
        # Chunky timestamp structure warning
        ts_diffs = np.diff(eeg['TS_UNIX'])
        if (ts_diffs == 0).mean() > 0.9:
            print("⚠️ Warning: >90% of timestamps are repeated (likely chunked data).")
        # Large gaps in data warning
        if (ts_diffs > np.timedelta64(2, 's')).any():
            print("⚠️ Warning: There are gaps >2 seconds in the timestamps.")

    return eeg

    # Extract client side timestamps from dedicated otree log columns
def get_actionlog_times(logs_df):
    # Get the timestamps from logging fields
    timestamps = logs_df.filter(regex='actions').dropna(axis=1, how='all').reset_index(drop=True)
    # display(timestamps)
    
    def extract_times(s):
        s = s[0].split(';') # Split string
        s = s[1:] # Remove first, empty entry
        # print(s)

        # Convert into DataFrame
        n_messages = int(len(s)/3)
        messages = []
        ts = []
        ts_unix = []
        for i in range(0, n_messages):
            step = i*3
            messages.append(s[step])
            ts.append(s[step+1])
            ts_unix.append(s[step+2])

        df = pd.DataFrame({'Message':messages, 
                           'TS':ts,
                           'TS_UNIX':ts_unix})

        # Convert unix timestamp to correct datetime
        df['TS'] = pd.to_datetime(pd.to_numeric(df['TS_UNIX'], errors='coerce'), unit='ms', utc=True).dt.tz_convert('Europe/Berlin')
        # print(df.Message.unique())

        return df

    # Parse TS cols
    # TODO: This code could be nicer using a split-apply-combine logic...
    timestamps_parsed = pd.DataFrame()
    for phase in timestamps.columns:
        msgs = extract_times(timestamps[phase])
        msgs['Exp_Phase'] = phase.replace('xTS', '')
        timestamps_parsed = pd.concat([timestamps_parsed, msgs], ignore_index=True) # Had to change this as append is deprecated in pandas since 2.0 -> Alternatively: versioning in requirements.txt 

        
    # Remove unnecessary events
    timestamps_parsed = timestamps_parsed[timestamps_parsed.Message.str.contains("taskStart|taskEnd")]
    # display(timestamps_parsed)
    # display(timestamps_parsed['Exp_Phase'].value_counts())
    
    # Restructure DF
    timestamps_parsed.drop(['TS_UNIX'], axis=1, inplace=True)
    timestamps_parsed['Message'] = timestamps_parsed['Message'].str.replace('task', '', regex=False) # Remove the substring "task"
    timestamps_parsed = timestamps_parsed.pivot(index='Exp_Phase', columns='Message', values='TS').reset_index() # Pivot the DataFrame
    
    timestamps_parsed[['app_name', 'round_number', 'player', 'page_name']] = timestamps_parsed['Exp_Phase'].str.split('.', expand=True)
    timestamps_parsed.drop(['Exp_Phase', 'player'], axis=1, inplace=True)
    timestamps_parsed['token'] = logs_df['participant.token'][0]
    
    # Add mapping to three EEG recordings
    mapping = {
        "_1": "rec1",
        "_2": "rec2",
        "_3": "rec3"
    }

    timestamps_parsed["rec"] = timestamps_parsed["app_name"].str[-2:].map(mapping)
    timestamps_parsed["app_name"] = timestamps_parsed["app_name"].str.replace(r"_[123]$", "", regex=True)
    # display(timestamps_parsed)
    
    # Structure IDs
    timestamps_parsed['page_name'] = timestamps_parsed['page_name'].replace('rest_actions_', '', regex=True)
    timestamps_parsed['page_name'] = timestamps_parsed['page_name'].replace('_actions', '', regex=True)
    timestamps_parsed['Exp_Phase'] = timestamps_parsed['app_name'] + "_" + timestamps_parsed['page_name'] + "_" + timestamps_parsed['round_number']
    
    return timestamps_parsed