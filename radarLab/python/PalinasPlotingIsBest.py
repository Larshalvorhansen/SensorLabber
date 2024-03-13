import sys
import numpy as np #Plotting
import matplotlib.pyplot as plt #Lese CSV
import csv
from scipy.fft import fft, ifft,fftfreq
# Plotting 
plt.rc('xtick', labelsize=19) # endre størrelsen på x-tall
plt.rc('ytick', labelsize=19) # endre størrelse på y-tall
plt.rcParams["figure.figsize"] = [8, 6] # endre størrelse på alle bildene 
#plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 2.5





#imortere fra ADC
def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.

    Returns sample period and a (`samples`, `channels`) `float64` array of
    sampled data from all channels.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype='uint16').astype('float64')
        # The "dangling" `.astype('float64')` casts data to double precision
        # Stops noisy autocorrelation due to overflow
        data = data.reshape((-1, channels))

    return sample_period, data 



#Plotte raw data til ADC
def plot_data(data, channels, filename='plot', xmin=0, xmax=1, sample_period=32e-6):
    """
    Plot raw data from ADC for specified channels.

    Parameters:
    - data (numpy.ndarray): The ADC data as a 2D array with shape (samples, channels).
    - channels (list): List of channels to plot.
    - filename (str, optional): Base name for saved plot files.
    - xmin, xmax (float, optional): X-axis limits for the combined plot.
    - sample_period (float, optional): Time between samples in seconds.
    """
    time = np.arange(data.shape[0]) * sample_period

    # Plot each channel in a separate subplot
    fig, axs = plt.subplots(len(channels), 1, figsize=(8, 2 * len(channels)), squeeze=False)
    for idx, channel in enumerate(channels):
        d = data[:, channel]
        axs[idx, 0].plot(time, d)
        axs[idx, 0].set_title(f'Channel {channel + 1}', fontsize=16)
        axs[idx, 0].set_xlabel('Time (s)', fontsize=16)
        axs[idx, 0].set_ylabel('Amplitude [mV]', fontsize=16)
        axs[idx, 0].grid()
        axs[idx, 0].set_xlim(xmin, xmax)

    plt.tight_layout()
    fig.savefig(f'{filename}_channels.png', dpi=300, bbox_inches='tight')

    # Plot all channels in one plot for comparison
    plt.figure(figsize=(22, 8))
    for channel in channels:
        plt.plot(time, data[:, channel] + channel*100, label=f'Channel {channel + 1}')  # Offset each channel for clarity

    plt.xlabel('Time (s)', fontsize=22)
    plt.ylabel('Amplitude [mV]', fontsize=22)
    plt.xlim(xmin, xmax)
    plt.grid()
    plt.legend(loc='best', fontsize='xx-large', frameon=True, shadow=True, borderpad=1)
    plt.savefig(f'{filename}_all_channels.png', dpi=300, bbox_inches='tight')


def calculate_SNR(positive_freq, signal_freq_range, noise_freq_range, positive_magnitude):
    """
    Calculate the Signal-to-Noise Ratio (SNR) within specified frequency ranges.

    Parameters:
    - positive_freq (np.ndarray): Array of positive frequencies.
    - signal_freq_range (tuple): The frequency range of the signal (start, end).
    - noise_freq_range (tuple): The frequency range of the noise (start, end).
    - positive_magnitude (np.ndarray): Array of magnitudes corresponding to `positive_freq`.

    Returns:
    - SNR_value (float): The calculated SNR in decibels.
    """
    signal_mask = (positive_freq >= signal_freq_range[0]) & (positive_freq <= signal_freq_range[1])
    noise_mask = (positive_freq >= noise_freq_range[0]) & (positive_freq <= noise_freq_range[1])

    signal_power = np.sum(positive_magnitude[signal_mask]**2)
    noise_power = np.sum(positive_magnitude[noise_mask]**2)

    # Prevent division by zero
    if noise_power == 0:
        return float('inf')  # Return infinity if noise power is zero

    SNR_value = 10 * np.log10(signal_power / noise_power)
    return SNR_value





def find_peak_frequency(start_freq, end_freq, positive_freq, positive_magnitude):
    """
    Identify the peak frequency and its magnitude within a specified frequency range.

    Parameters:
    - start_freq (float): The start of the frequency range of interest.
    - end_freq (float): The end of the frequency range of interest.
    - positive_freq (np.ndarray): Array of positive frequencies.
    - positive_magnitude (np.ndarray): Array of magnitudes corresponding to `positive_freq`.

    Returns:
    - A tuple (peak_frequency, peak_magnitude) with the peak frequency in Hz and its magnitude.
    """
    freq_range_mask = (positive_freq >= start_freq) & (positive_freq <= end_freq)
    freq_in_range = positive_freq[freq_range_mask]
    magnitude_in_range = positive_magnitude[freq_range_mask]

    if not magnitude_in_range.size:
        return (None, None)  # Return None if no data is in the specified range

    max_index = np.argmax(magnitude_in_range)
    peak_frequency = freq_in_range[max_index]
    peak_magnitude = magnitude_in_range[max_index]

    return peak_frequency, peak_magnitude


def calculate_fft_with_zero_padding(data, sample_rate, frec_spek, signal_freq_range, noise_freq_range, Title="Bilde1"):
    """
    Calculates the FFT (Fast Fourier Transform) with zero-padding for each channel in the input data,
    identifies the peak frequency and its magnitude, and computes the Signal-to-Noise Ratio (SNR) for a specified frequency range.
    
    Parameters:
    - data (numpy.ndarray): The input signal data, expected to be a 2D array where each column represents a channel.
    - sample_rate (float): The sampling rate of the data in Hz.
    - frec_spek (float): The frequency range for consideration in Hz.
    - signal_freq_range (tuple): A tuple indicating the frequency range considered as the signal for SNR calculation.
    - noise_freq_range (tuple): A tuple indicating the frequency range considered as noise for SNR calculation.
    - Title (str, optional): Title for the operation (not used in the output but kept for compatibility).

    Returns:
    - SNR_value (float): The Signal-to-Noise Ratio of the last processed channel in dB.
    - frequency_topp (float): The peak frequency in Hz of the last processed channel.
    - magnitude_topp (float): The magnitude of the peak frequency for the last processed channel.
    
    Notes:
    - The function prints the peak frequency and its magnitude for each channel processed.
    - SNR calculation depends on an external `calculate_SNR` function.
    - Peak frequency identification depends on an external `find_peak_frequency` function.
    """
    
    channels=data.shape[1]

    for j in range(channels):
        d = data[:, j]
        d = d - np.mean(d)

        # Apply Hann window to the signal
        windowed_data = d * np.hanning(len(d))

        # Zero-padding: Length to the next power of 2 for better FFT performance and resolution
        N = len(windowed_data)
        N_padded = 2**np.ceil(np.log2(N)).astype(int)

        # Perform FFT with zero-padding
        fft_result = fft(windowed_data, n=N_padded)
        fft_magnitude = np.abs(fft_result)

        # Frequency bins
        freq = fftfreq(N_padded, 1/sample_rate)

        # Adjusted to use the full spectrum for plotting and calculations
        full_freq = freq
        full_magnitude = fft_magnitude

        # Calculate SNR - might still focus on a part of the spectrum for relevance
        SNR_value = calculate_SNR(full_freq[:N_padded//2], signal_freq_range, noise_freq_range, full_magnitude[:N_padded//2])
        frequency_topp, magnitude_topp = find_peak_frequency(-frec_spek, frec_spek, full_freq, full_magnitude)

        #print(f"Peak frequency: {frequency_topp} Hz, Peak magnitude: {magnitude_topp/10000} dB")
        #print(f'Channel {j+1} SNR: {SNR_value:.2f} dB')

    return SNR_value, frequency_topp, magnitude_topp



def plot_fft_with_zero_padding(data, sample_rate, frec_spek, signal_freq_range, noise_freq_range, Title="Bilde1"):
    """
    Plots the FFT (Fast Fourier Transform) of each channel in the data with zero-padding, 
    including applying a window function to the signal to reduce edge effects and improve FFT performance.
    
    Parameters:
    - data (numpy.ndarray): Input signal data, expected to be a 2D array where each column represents a channel.
    - sample_rate (float): The sampling rate of the data in Hz.
    - frec_spek (float): The frequency range for the x-axis of the plot in Hz.
    - signal_freq_range (tuple): A tuple indicating the frequency range considered as the signal for SNR calculation.
    - noise_freq_range (tuple): A tuple indicating the frequency range considered as noise for SNR calculation.
    - Title (str, optional): Title of the plot. Defaults to "Bilde1".

    Outputs:
    - A plot displaying the FFT magnitude in dB for each channel within the specified frequency range.
    """
    plt.figure(figsize=(22, 8))
    channels=data.shape[1]

    for j in range(channels):
        d = data[:, j]
        d = d - np.mean(d)

        # Apply Hann window to the signal
        windowed_data = d * np.hanning(len(d))

        # Zero-padding: Length to the next power of 2 for better FFT performance and resolution
        N = len(windowed_data)
        N_padded = 2**np.ceil(np.log2(N)).astype(int)

        # Perform FFT with zero-padding
        fft_result = fft(windowed_data, n=N_padded)
        fft_magnitude = np.abs(fft_result)

        # Frequency bins
        freq = fftfreq(N_padded, 1/sample_rate)

        # Adjusted to use the full spectrum for plotting and calculations
        full_freq = freq
        full_magnitude = fft_magnitude

        plt.plot(full_freq, 20*np.log10(full_magnitude) - np.max(20*np.log10(full_magnitude)), label=f'Channel {j+1}')    
    plt.xlabel('Frequency (Hz)', fontsize=22)
    plt.ylabel('Magnitude (dB)', fontsize=22)
    plt.xlim(-frec_spek, frec_spek)
    plt.ylim(np.min(20*np.log10(full_magnitude))-100,5)  # Adjust y-axis limits appropriately
    plt.grid(True)
    plt.title(Title)
    plt.legend(loc='best', fontsize='xx-large', frameon=True, shadow=True, borderpad=1)
    plt.show()


#Testing 

#plot_data(data,(2,4), 'channels',0.2,0.25)
frec_spek = 1000
signal_freq_range = (980, 1010)
noise_freq_range = (1100, 1200) 

hastighetmot = ['data/hastighetmotsat1','data/hastighetmotsat2','data/hastighetmotsatt21','data/hastighetmotsatt22','data/hastighetmotsatt23','data/hastighetmotsatt24','data/hastighetmotsatt25','data/hastighetmotsatt26','data/hastighetmotsatt27']
hastighet1 =['data/hastighet11','data/hastighet12', 'data/hastighet13','data/hastighet14','data/hastighet15','data/hastighet16']
hastighet2 =['data/hastighet21','data/hastighet22', 'data/hastighet23','data/hastighet24','data/hastighet25','data/hastighet26','data/hastighet27']



hast_mot_data=[]
hast_mot_peaks=[]
SNR_mot_peaks=[]
# Loop through all files
for filename in hastighetmot:
    # Read data using your custom function (assuming it returns two values)
    sample_rate, data_mot = raspi_import(filename)
    data_mot = np.array([data_mot[:,2]+1j*data_mot[:,4]]).T
    # Append the data to your storage list
    hast_mot_data.append((sample_rate, data_mot))
    
    # Compute FFT and find peaks (adjust the arguments as necessary)
    #plot_fft_with_zero_padding(data_mot , 31250, frec_spek,signal_freq_range,noise_freq_range,'Motsatt hastighet')
    SNR, peak_f_result, max_magnitude= calculate_fft_with_zero_padding(data_mot , 31250, frec_spek,signal_freq_range,noise_freq_range,'Motsatt hastighet')
    # Store the peaks
    hast_mot_peaks.append(peak_f_result)
    SNR_mot_peaks.append(SNR)
    #print(f"Peak frequency: {frequency_topp} Hz, Peak magnitude: {magnitude_topp/10000} dB")
    #print(f' Motsatt hastighet {1} SNR: {SNR:.2f} dB')




hast_1_data=[]
hast_1_peaks=[]
SNR_1_peaks=[]
for filename in hastighet1:
    # Read data using your custom function (assuming it returns two values)
    sample_rate, data_1 = raspi_import(filename)
    data_1 = np.array([data_1[:,2]+1j*data_1[:,4]]).T
    # Append the data to your storage list
    hast_1_data.append((sample_rate, data_1))
    
    # Compute FFT and find peaks (adjust the arguments as necessary)
    SNR, peak_f_result, max_magnitude= calculate_fft_with_zero_padding(data_1 , 31250, frec_spek,signal_freq_range,noise_freq_range,'Hastighet 1')
    # Store the peaks
    hast_1_peaks.append(peak_f_result)
    SNR_1_peaks.append(SNR)

hast_2_data=[]
hast_2_peaks=[]
SNR_2_peaks=[]
for filename in hastighet2:
    # Read data using your custom function (assuming it returns two values)
    sample_rate, data_2 = raspi_import(filename)
    data_2 = np.array([data_2[:,2]+1j*data_2[:,4]]).T
    # Append the data to your storage list
    hast_2_data.append((sample_rate, data_2))
    
    # Compute FFT and find peaks (adjust the arguments as necessary)
    SNR, peak_f_result, max_magnitude= calculate_fft_with_zero_padding(data_2, 31250, frec_spek,signal_freq_range,noise_freq_range,'Hastighet 2')

    # Store the peaks
    hast_2_peaks.append(peak_f_result)
    SNR_2_peaks.append(SNR)

print("Frekvens til hastighet i motsatt retning", hast_mot_peaks,"\n")
print("Frekvens til hastighet 1",hast_1_peaks,"\n")
print("Frekvens til hastighet 2",hast_2_peaks,"\n")


