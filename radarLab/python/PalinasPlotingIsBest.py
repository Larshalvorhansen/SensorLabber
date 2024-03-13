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
def plot_data(data,channels,filename='plot',xmin = 0,xmax= 1):
    # Number of channels
    num_channels = data.shape[1]
    
    #num_channels = [0,1,2,4]
    # Time array (assuming equal spacing)
    sample_period = 32e-6  # example value, replace with your actual sample period
    time = np.arange(data.shape[0]) * sample_period
     
    
     
    # Plot each channel in a separate subplot
    fig, axs = plt.subplots(num_channels, 1, figsize=(8, 10))
    #for i in range(num_channels):
    for i in(channels):
        print (i)
        d=data[:, i]
        axs[i].plot(time, d)
        axs[i].set_title(f'Channel {i+1}',fontsize=16)
        axs[i].set_xlabel('Time (s)',fontsize=16)
        axs[1].set_ylabel('Amplitude [mV]', fontsize=16)
        axs[i].grid()
        axs[i].set_xlim(0.2, 0.4)
    plt.tight_layout()
    fig.savefig(f'{filename}_channels.png', dpi=300, bbox_inches='tight')

    # Plot all channels in one plot for comparison
    plt.figure(figsize=(22, 8))
    for i in (channels):
        plt.plot(time, data[:, i]+i, label=f'Channel {i+1}')

    plt.xlabel('Time (s)',fontsize=22)
    plt.ylabel('Amplitude [mV]',fontsize=22)
    #plt.title('All Channels',fontsize=16)
    plt.xlim(xmin, xmax)
    plt.grid()
    plt.legend(loc='best', fontsize='xx-large', frameon=True, shadow=True, borderpad=1)
    plt.savefig(f'{filename}_all_channels.png', dpi=300, bbox_inches='tight')


# FFT og ovs ....
    
def calculate_SNR(positive_freq, signal_freq_range, noise_freq_range, positive_magnitude):
    # Calculate power in the signal and noise frequency ranges
    signal_mask = (positive_freq >= signal_freq_range[0]) & (positive_freq <= signal_freq_range[1])
    noise_mask = (positive_freq >= noise_freq_range[0]) & (positive_freq <= noise_freq_range[1])

    signal_power = np.sum(positive_magnitude[signal_mask]**2)
    noise_power = np.sum(positive_magnitude[noise_mask]**2)

    # Calculate SNR
    SNR_value = 10 * np.log10(signal_power / noise_power)
    return SNR_value




def peak (min, max, positive_freq, positive_magnitude):
        # Define your frequency range
        start_freq = min
        end_freq = max

        # Step 1: Identify the indices within the specified frequency range
        freq_range_mask = (positive_freq >= start_freq) & (positive_freq <= end_freq)

        # Apply the mask to get frequencies and magnitudes within the range
        freq_in_range = positive_freq[freq_range_mask]
        magnitude_in_range = positive_magnitude[freq_range_mask]

        # Step 2: Find the index of the maximum value in the magnitude within the range
        max_index = np.argmax(magnitude_in_range)

        # Step 3: Extract the frequency and magnitude of the peak
        peak_frequency = freq_in_range[max_index]
        peak_magnitude = magnitude_in_range[max_index]

        print(f"Peak frequency: {peak_frequency} Hz, Peak magnitude: {peak_magnitude} dB")
        return peak_frequency, peak_magnitude



"""

def plot_fft_with_zero_padding(data, sample_rate, frec_spek, signal_freq_range, noise_freq_range,channels,Title="Bilde1"):
    """
"""
    Plot the FFT of multiple signals with zero-padding and Hann window applied and calculate SNR.

    Parameters:
    data (numpy.ndarray): The input signals, expected shape (samples, channels).
    sample_rate (float): The sampling rate of the signals.
    frec_spek (float): The maximum frequency to be plotted.
    signal_freq_range (tuple): The frequency range considered as signal (start_freq, end_freq).
    noise_freq_range (tuple): The frequency range considered as noise (start_freq, end_freq).
    """
"""
    plt.figure(figsize=(22, 8))
    
    #for j in range(data.shape[1]):  # Iterate over channels
    for j in (channels):
        d = data[:, j]
        d = d - np.mean(d)

        # Apply Hann window to the signal
        windowed_data = d * np.hanning(len(d))

        # Zero-padding: Length to the next power of 2 for better FFT performance and resolution
        N = len(windowed_data)
        #N_padded=N
        N_padded = 2**np.ceil(np.log2(N)).astype(int)

        # Perform FFT with zero-padding
        fft_result = fft(windowed_data, n=N_padded)
        fft_magnitude = np.abs(fft_result)

        # Frequency bins
        freq = fftfreq(N_padded, 1/sample_rate)

        # Only take the positive half of the spectrum
        positive_freq = freq[:N_padded//2]
        positive_magnitude = fft_magnitude[:N_padded//2]

        
        SNR_value = calculate_SNR(positive_freq, signal_freq_range, noise_freq_range, positive_magnitude)
        print(f'Channel {j+1} SNR: {SNR_value:.2f} dB')
        # Plotting each frequency component in the same figure
        #plt.plot(positive_freq, 20*np.log10(positive_magnitude), label=f'Channel {j+1} - SNR: {SNR:.2f} dB')  # Plot in dB
        plt.plot(positive_freq, 20*np.log10(positive_magnitude) - np.max(20*np.log10(positive_magnitude)), label=f'Channel {j+1}')  # Convert magnitude to dB
   
   # Assuming 'positive_freq' and 'positive_magnitude' are your frequency and magnitude arrays
    

    print(np.min(20*np.log10(positive_magnitude))-5)
    plt.xlabel('Frequency (Hz)', fontsize=22)
    plt.ylabel('Magnitude (dB)', fontsize=22)
    plt.xlim(-frec_spek, frec_spek)
    plt.ylim(np.min(20*np.log10(positive_magnitude))-100,5)  # Adjust the y-axis limits appropriately
    plt.grid(True)
    plt.title(Title)
    plt.legend(loc='best', fontsize='xx-large', frameon=True, shadow=True, borderpad=1)
    plt.show()


    return peak (3,1000, positive_freq, positive_magnitude)



"""




def plot_fft_with_zero_padding(data, sample_rate, frec_spek, signal_freq_range, noise_freq_range, Title="Bilde1"):
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

        # Calculate SNR - might still focus on a part of the spectrum for relevance
        SNR_value = calculate_SNR(full_freq[:N_padded//2], signal_freq_range, noise_freq_range, full_magnitude[:N_padded//2])
        print(f'Channel {j+1} SNR: {SNR_value:.2f} dB')

        plt.plot(full_freq, 20*np.log10(full_magnitude) - np.max(20*np.log10(full_magnitude)), label=f'Channel {j+1}')

    plt.xlabel('Frequency (Hz)', fontsize=22)
    plt.ylabel('Magnitude (dB)', fontsize=22)
    plt.xlim(-frec_spek, frec_spek)
    plt.ylim(np.min(20*np.log10(full_magnitude))-100,5)  # Adjust y-axis limits appropriately
    plt.grid(True)
    plt.title(Title)
    plt.legend(loc='best', fontsize='xx-large', frameon=True, shadow=True, borderpad=1)
    plt.show()

    # Depending on your requirements, adjust or keep this call to focus on the positive spectrum for peak analysis
    return peak(3, 1000, full_freq[:N_padded//2], full_magnitude[:N_padded//2])


#Testing 

sample_rate, data = raspi_import('data/hastighet11')
print(data.shape,sample_rate )

print(data)
frec_spek = 1300

data = np.array([data[:,2]+1j*data[:,4]]).T  # Makes data two-dimensional again but with one "channel"

#plot_data(data,(2,4), 'channels',0.2,0.25)
signal_freq_range = (980, 1010)
noise_freq_range = (1100, 1200) 
#plot_fft_with_zero_padding(data, sample_rate, frec_spek, signal_freq_range, noise_freq_range)
#plot_fft_without_window_or_padding(data, sample_rate, frec_spek)
#plot_fft_with_zero_padding_1(data, 31250, frec_spek)
plot_fft_with_zero_padding(data, 31250, frec_spek,signal_freq_range,noise_freq_range)
