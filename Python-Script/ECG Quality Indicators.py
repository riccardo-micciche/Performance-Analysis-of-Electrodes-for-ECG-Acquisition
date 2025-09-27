import matplotlib
matplotlib.use('Qt5Agg') # To visualize interactive plots
import numpy as np

from Functions import std_calc, ptp_calc, find_waves, import_data
from scipy.stats import kurtosis

# Load the in-vivo measurement data from the .csv file
raw = import_data(r"filepath")

# Separate the data into isoelectric segments and PQRST complexes
raw_iso, raw_pqrst, isoelectric_segments, pqrst_complexes = find_waves(raw)

# Calculate the median standard deviation of the isoelectric segments
median_rms = std_calc(isoelectric_segments)

# Calculate the median peak-to-peak value across the PQRST complexes
median_ptp = ptp_calc(pqrst_complexes)

# Compute the SNR
SNR = round(20*np.log10(median_ptp/(4*median_rms)), 2)

# Compute the Kurtosis
k = round(kurtosis(raw), 2)

# Print the results
print(
    f"The median standard deviation of the isoelectric segments is {median_rms}\n"
    f"The median peak-to-peak amplitude across the PQRST complexes is {median_ptp}\n"
    f"The SNR is {SNR}\n"
    f"The Kurtosis is {k}"
)