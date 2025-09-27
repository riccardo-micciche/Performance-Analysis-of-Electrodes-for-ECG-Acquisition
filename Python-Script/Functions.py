import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def import_data(filepath: str):
    data = pd.read_csv(filepath, skiprows = 2) # skiprows is necessary due to the formatting of data obtained through the MEMS Studio GUI
    raw = -1 * data["vAFE[LSB]"] / 1311 # -1 is present due to how data was acquired, if not necessary this line should be commented

    return raw

def find_waves(raw):
    # Clean the signal so that the PQRST complexes can be identified more easily
    clean = nk.ecg_clean(raw, sampling_rate=800, method='neurokit')

    # Identify the position of the PQRST complexes
    _, info_R = nk.ecg_peaks(clean, sampling_rate=800, method='nabian2018')
    r_peaks = info_R["ECG_R_Peaks"]
    _, waves = nk.ecg_delineate(clean, r_peaks, sampling_rate=800, method="prominence")
    t_off = waves["ECG_T_Offsets"]
    r_to_r = np.diff(r_peaks)
    # Since the ECG is acquired far from the chest, the amplitude of P waves is too small to reliably identify their location,
    # making it impossible to use the onset of a P wave to determine the end of an isoelectric segment. To circumvent this issue,
    # the segments were instead delimited by estimating their length as a portion of the distance between the R peak of the PQRST
    # complex under consideration and the following R peak.

    # Initialize the lists required to store the desired segments
    pqrst_segments = []
    isoelectric_segments = []
    start_cut = t_off[0] + np.floor(r_to_r[0]/15)
    end_cut = t_off[-1]
    raw_cut = raw.loc[start_cut:end_cut].copy() # The signal is cut to begin with an isoelectric segment and end with a PQRST complex
    raw_iso = raw_cut.copy()
    raw_pqrst = raw_cut.copy()

    # Select the parts to separate
    for x, d in zip(t_off[:-1], r_to_r):
        start = x + np.floor(d/15)
        end = x + np.ceil(d/2.5)
        isoelectric_segments.append(raw.loc[start:end]) # Store each isoelectric segments in an array for later processing
        raw_pqrst.loc[start:end] = np.nan # Generate an array with the PQRST complexes that can be plot for a visual check
    raw_iso = raw_iso.where(raw_pqrst.isna()) # Generate an array with the isoelectric segments that can be plot for a visual check

    # Generate PQRST segments as contiguous non-NaN runs
    mask = raw_pqrst.notna()
    groups = (mask.ne(mask.shift())).cumsum()
    for _, seg in raw_pqrst[mask].groupby(groups[mask]):
        pqrst_segments.append(seg) # Store each PQRST complex in an array for later processing

    # Plot raw, the isoelectric lines and the pqrst segments
    triple_plot(raw_cut, raw_iso, raw_pqrst) # Plotting the original signal and the segmentation provides a visual sanity check of the processing

    return raw_iso, raw_pqrst, isoelectric_segments, pqrst_segments

def triple_plot(raw_cut, iso_segs, pqrst_segs):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(raw_cut, color='blue')
    axs[0].set_title('ECG')
    axs[0].set_xlabel('Sample Index')
    axs[0].set_ylabel('Amplitude [mV]')
    # axs[0].set_ylim(1.5, 3) # Limits can be set to visualize the isoelectric segments in the same scale as the PQRST complexes
    axs[0].grid(True)
    axs[1].plot(iso_segs, color='orange')
    axs[1].set_title('Isoelectric segments')
    axs[1].set_xlabel('Sample Index')
    axs[1].set_ylabel('Amplitude [mV]')
    # axs[1].set_ylim(1.5, 3)
    axs[1].grid(True)
    axs[2].plot(pqrst_segs, color='blue')
    axs[2].set_title('PQRST complexes')
    axs[2].set_xlabel('Sample Index')
    axs[2].set_ylabel('Amplitude [mV]')
    # axs[2].set_ylim(1.5, 3)
    axs[2].grid(True)
    plt.tight_layout()
    plt.show()

def std_calc(segments):
    # Calculate the standard deviation of each isoelectric segment. Std is used instead of RMS since the calculations are carried out on an unfiltered signal
    # which, unlike the ECG itself, is not centered at 0 V.
    std_values = []
    for segment in segments:
        std = round(np.std(segment), 4)
        std_values.append(std)
    seg_std = np.array(std_values)

    # Calculate the median of the std values
    median_std = round(np.median(seg_std), 4)

    return median_std

def ptp_calc(segments):
    # Calculate the peak to peak (ptp) amplitude of each segment
    ptp_values = []
    for segment in segments:
        ptp = np.max(segment) - np.min(segment)
        ptp_values.append(ptp)
    seg_ptp = np.array(ptp_values)

    # Calculate the median of the peak to peak values
    median_ptp = round(np.median(seg_ptp), 4)

    return median_ptp