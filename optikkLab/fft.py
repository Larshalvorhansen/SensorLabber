import sys
import numpy as np #Plotting
import matplotlib.pyplot as plt #Lese CSV
import csv
from scipy.signal import detrend
from scipy.fft import fft, ifft,fftfreq
# Plotting 
plt.rc('xtick', labelsize=19) # endre størrelsen på x-tall
plt.rc('ytick', labelsize=19) # endre størrelse på y-tall
plt.rcParams["figure.figsize"] = [8, 6] # endre størrelse på alle bildene 
#plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 2.5



def raspi_import(path, channels=3):
    """
    Import data produced using adc_sampler.c.
    
    Parameters:
    - path: str, path to the data file.
    - channels: int, number of data channels. Default is 3.

    Returns:
    - sample_period: int, sample period in ms.
    - data: numpy.ndarray, a (`samples`, `channels`) array of sampled data from all channels.
    """
    
    try:
        # Use with statement for safe file handling
        with open(path, 'r') as fid:
            # Use numpy to efficiently load data into an array
            data = np.loadtxt(fid, delimiter=' ')
            
            # Validate the shape of the data if necessary
            if data.shape[1] != channels:
                raise ValueError(f"Expected data to have {channels} channels, but found {data.shape[1]}")
    
    except IOError:
        print(f"Error: File {path} could not be opened.")
        return None, None
    except ValueError as e:
        print(f"Error: {e}")
        return None, None

    # Example way to dynamically determine the sample period
    # This is placeholder logic; adjust based on your actual data format and needs
    sample_period = 30

    return sample_period, data


def plot_data(data, sample_period=1/30, filename='plot', separate_channels=False):
    """
    Plot data from multiple channels.
    
    Parameters:
    - data: numpy.ndarray, the data to plot with shape (`samples`, `channels`).
    - sample_period: float, sample period in seconds. Default is 1/30.
    - filename: str, base name for the output file.
    - separate_channels: bool, whether to plot each channel in a separate subplot.
    """
    channels_names = ['Blå fargekanal', 'Grønn fargekanal', 'Rød fargekanal']
    channels__colors= ['b', 'g', 'r']
    num_channels = data.shape[1]  # Determine the number of channels dynamically
    time = np.arange(data.shape[0]) * sample_period

    if separate_channels:
        # Plot each channel in a separate subplot
        fig, axs = plt.subplots(num_channels, 1, figsize=(8, num_channels*2))
        for i in range(num_channels):
            axs[i].plot(time, data[:, i], channels__colors[i],label = channels_names[i])
            axs[i].set_title(f'Channel {i+1}')
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel('Amplitude [mV]')
            axs[i].grid()
            axs[i].legend(loc='best', fontsize='xx-large')
     
        plt.tight_layout()
        #fig.savefig(f'{filename}_channels.png', dpi=300, bbox_inches='tight')
    else:
        # Plot all channels in one plot for comparison
        plt.figure(figsize=(22, 8))
        for i in range(num_channels):
            plt.plot(time, data[:, i], channels__colors[i],label=channels_names[i])
            plt.xlabel('Time (s)', fontsize=22)
            plt.ylabel('Amplitude [mV]', fontsize=22)
            plt.grid()
            plt.legend(loc='best', fontsize='xx-large')
        #plt.savefig(f'{filename}_all_channels.png', dpi=300, bbox_inches='tight')

    plt.show()


def calculate_SNR(positive_freq, positive_magnitude, signal_freq_range, noise_freq_range,):
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


def find_peak_frequency(start_freq, end_freq, freq, magnitude):
    """
    Identify the peak frequency and its magnitude within a specified frequency range.

    Parameters:
    - start_freq (float): The start of the frequency range of interest.
    - end_freq (float): The end of the frequency range of interest.
    - freq (np.ndarray): Array of frequencies.
    - magnitude (np.ndarray): Array of magnitudes corresponding to `freq`.

    Returns:
    - A tuple (peak_frequency, peak_magnitude) with the peak frequency in Hz and its magnitude.
    """
    freq_range_mask = (freq >= start_freq) & (freq <= end_freq)
    freq_in_range = freq[freq_range_mask]
    magnitude_in_range = magnitude[freq_range_mask]

    if not magnitude_in_range.size:
        return (None, None)  # Return None if no data is in the specified range

    max_index = np.argmax(magnitude_in_range)
    peak_frequency = freq_in_range[max_index]
    peak_magnitude = magnitude_in_range[max_index]

    return peak_frequency, peak_magnitude


def calculate_fft_with_zero_padding(data, sample_rate, frec_spek):
    """
    Calculates the FFT (Fast Fourier Transform) with zero-padding for each channel in the input data,
    identifies the peak frequency and its magnitude, and computes the Signal-to-Noise Ratio (SNR) for a specified frequency range.
    
    Parameters:
    - data (numpy.ndarray): The input signal data, expected to be a 2D array where each column represents a channel.
    - sample_rate (float): The sampling rate of the data in Hz.
    - frec_spek (float): The frequency range for consideration in Hz.

    Returns:
    - SNRs (numpy.ndarray): An array of the Signal-to-Noise Ratios for each channel in dB.
    - frequency_topps (numpy.ndarray): An array of the peak frequencies in Hz for each channel.
    - magnitude_topps (numpy.ndarray): An array of the magnitudes of the peak frequencies for each channel.
    
    Notes:
    - The function prints the peak frequency and its magnitude for each channel processed.
    - SNR calculation depends on an external `calculate_SNR` function.
    - Peak frequency identification depends on an external `find_peak_frequency` function.
    """
    
    channels = data.shape[1]
    SNRs = np.zeros(channels)
    frequency_topps = np.zeros(channels)
    magnitude_topps = np.zeros(channels)

    for j in range(channels):
        d = data[:, j] - np.mean(data[:, j])  # Subtract the mean to center the signal

        # Apply Hann window to the signal
        
        windowed_data = d * np.hanning(len(d))

        # Zero-padding: Length to the next power of 2 for better FFT performance and resolution
        N_padded = 2**np.ceil(np.log2(len(windowed_data))).astype(int)

        # Perform FFT with zero-padding
        fft_result = fft(windowed_data, n=N_padded)
        fft_magnitude = np.abs(fft_result[:N_padded//2])  # Only use positive frequencies

        freq = fftfreq(N_padded, 1/sample_rate)[:N_padded//2]

        # find_peak_frequency returns the peak frequency and its magnitude
        frequency_topp, magnitude_topp = find_peak_frequency(0.5, frec_spek, freq, fft_magnitude)        
        signal_freq_range = (frequency_topp-0.5, frequency_topp+0.5)
        noise_freq_range = (frequency_topp+0.5, frequency_topp+5)

        SNRs[j] = calculate_SNR(freq, fft_magnitude, signal_freq_range, noise_freq_range)
        frequency_topps[j] = frequency_topp
        magnitude_topps[j] = magnitude_topp

    return SNRs, frequency_topps, magnitude_topps


def plot_fft_with_zero_padding(data, sample_rate, frec_spek, Title="Bilde1", full = 0):
    """
    Plots the FFT (Fast Fourier Transform) of each channel in the data with zero-padding, 
    including applying a window function to the signal to reduce edge effects and improve FFT performance.
    
    Parameters:
    - data (numpy.ndarray): Input signal data, expected to be a 2D array where each column represents a channel.
    - sample_rate (float): The sampling rate of the data in Hz.
    - frec_spek (float): The frequency range for the x-axis of the plot in Hz.
    - Title (str, optional): Title of the plot. Defaults to "Bilde1".

    Outputs:
    - A plot displaying the FFT magnitude in dB for each channel within the specified frequency range.
    """
    plt.figure(figsize=(22, 8))
    channels=data.shape[1]
    channels_names = ['Blå', 'Grønn', 'Rød']
    channels__colors= ['b', 'g', 'r']
    for j in range(channels):
        d = data[:, j]
        d = d - np.mean(d)
        d = detrend(d)

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

        frequency_topp, magnitude_topp = find_peak_frequency(0.5, frec_spek, full_freq, full_magnitude) 

        # For å enklere plotte både negative og positive frekvenser 
        if full == 0:
            full_freq = freq [:N_padded//2]
            full_magnitude = fft_magnitude [:N_padded//2]

        plt.plot(full_freq, 20*np.log10(full_magnitude) - np.max(20*np.log10(full_magnitude)), channels__colors[j],label=channels_names[j])
        plt.scatter(frequency_topp, magnitude_topp, color="black", marker='o', s=100)  # `s` is the size of the marker


        
    plt.xlabel('Frequency (Hz)', fontsize=22)
    plt.ylabel('Magnitude (dB)', fontsize=22)
    plt.xlim(0.5, frec_spek)
    plt.ylim(np.min(20*np.log10(full_magnitude))-70,5)  # Adjust y-axis limits appropriately
    plt.grid(True)
    plt.title(Title)
    plt.legend(loc='best', fontsize='xx-large', frameon=True, shadow=True, borderpad=1)
    plt.show()






#Testing
    
ok = ['data_num/ok1','data_num/ok2','data_num/ok4','data_num/ok5','data_num/ok6']
lille = ['data_num/lille2','data_num/lille3','data_num/lille4']
palina =['data_num/palina_r1','data_num/palina_r2','data_num/palina1','data_num/palina2']
random = ['data_num/test_lab','data_num/robust4','data_num/test_frekvens','data_num/test_out']

frec_spek = 5


def test(file_list,plot_fft=0,plot_data_flag=0):
    """
    Test and plot FFT data for a list of files, organized by channel.

    Parameters:
    - file_list: list of file names.
    """

    channel_colors = ['Blå', 'Rød', 'Grønn']  # Example channel names/colors
    results = {'Blå': {'Peaks': [], 'SNR': [], 'Puls': []},
               'Rød': {'Peaks': [], 'SNR': [], 'Puls': []},
               'Grønn': {'Peaks': [], 'SNR': [], 'Puls': []}}

    for filename in file_list:
        # Assuming raspi_import returns sample_rate and data
        sample_rate, data_test = raspi_import(filename)
        if plot_fft==1:   
            plot_fft_with_zero_padding(data_test, 30, frec_spek,f"Spektrum plot: {filename}")
        if plot_data_flag==1:
            plot_data(data_test, filename=f"Raw data plot:{filename}",separate_channels=True)

        # Assuming calculate_fft_with_zero_padding returns SNRs, peaks, magnitudes for all channels
        SNRs, peak_results, magnitudes = calculate_fft_with_zero_padding(data_test, sample_rate, frec_spek)

        # Process results for each channel
        for i, color in enumerate(channel_colors):
            results[color]['Peaks'].append(peak_results[i])
            results[color]['SNR'].append(SNRs[i])
            results[color]['Puls'].append(peak_results[i] * 60)  # Convert peak frequency to beats per minute

    # Print results in a more organized way
    for color in channel_colors:
        print(f"Kanal {color}:")
        print(f"  Sterkeste frekvensen: {results[color]['Peaks']}")
        print(f"  SNR: {results[color]['SNR']}")
        print(f"  Puls: {results[color]['Puls']}\n")

# Example usage
        

test(ok,1)


