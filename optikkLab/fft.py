import sys
import numpy as np #Plotting
import matplotlib.pyplot as plt #Lese CSV
import csv
from scipy.signal import correlate
from scipy.fft import fft, ifft,fftfreq
# Plotting 
plt.rc('xtick', labelsize=19) # endre størrelsen på x-tall
plt.rc('ytick', labelsize=19) # endre størrelse på y-tall
plt.rcParams["figure.figsize"] = [8, 6] # endre størrelse på alle bildene 
#plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 2.5



def raspi_import(path, channels=3):

    fid = open(path, 'r')
    data = []
    for line in fid:
        values = line.split(" ")
        temp = []
        for num in values:
            temp.append(float(num))
        data.append(temp)
    #print(data)
    data = np.array(data)
    return 30, data

def raspi_import_1(path, channels=3):
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
        print(data)

    return sample_period, data 


def plot_data(data,filename='plot'):
    # Number of channels
    num_channels = 3
    
    # Time array (assuming equal spacing)
    sample_period = 1/30  # example value, replace with your actual sample period
    time = np.arange(data.shape[0]) * sample_period

    """# Plot each channel in a separate subplot
    fig, axs = plt.subplots(num_channels, 1, figsize=(8, 10))
    for i in range(num_channels):
        d=data[:, i]
        axs[i].plot(time, d)
        axs[i].set_title(f'Channel {i+1}',fontsize=16)
        axs[i].set_xlabel('Time (s)',fontsize=16)
        axs[1].set_ylabel('Amplitude [mV]', fontsize=16)
        axs[i].grid()
        axs[i].set_xlim(0.2, 0.4)
    plt.tight_layout()
    fig.savefig(f'{filename}_channels.png', dpi=300, bbox_inches='tight')
"""
    # Plot all channels in one plot for comparison
    plt.figure(figsize=(22, 8))
    for i in range(num_channels):
        plt.plot(time, data[:, i], label=f'Channel {i+1}')

    plt.xlabel('Time (s)',fontsize=22)
    plt.ylabel('Amplitude [mV]',fontsize=22)
    #plt.title('All Channels',fontsize=16)
    plt.xlim(0, 30)
    plt.grid()
    plt.legend(loc='best', fontsize='xx-large', frameon=True, shadow=True, borderpad=1)
    plt.savefig(f'{filename}_all_channels.png', dpi=300, bbox_inches='tight')


"""
def plot_fft_with_zero_padding(data, sample_rate, frec_spek, signal_freq_range, noise_freq_range):
    
    Plot the FFT of multiple signals with zero-padding and Hann window applied and calculate SNR.

    Parameters:
    data (numpy.ndarray): The input signals, expected shape (samples, channels).
    sample_rate (float): The sampling rate of the signals.
    frec_spek (float): The maximum frequency to be plotted.
    signal_freq_range (tuple): The frequency range considered as signal (start_freq, end_freq).
    noise_freq_range (tuple): The frequency range considered as noise (start_freq, end_freq).
    

    plt.figure(figsize=(22, 8))

    for j in range(data.shape[1]):  # Iterate over channels
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

        # Only take the positive half of the spectrum
        positive_freq = freq[:N_padded//2]
        positive_magnitude = fft_magnitude[:N_padded//2]

        # Calculate power in the signal and noise frequency ranges
        signal_mask = (positive_freq >= signal_freq_range[0]) & (positive_freq <= signal_freq_range[1])
        noise_mask = (positive_freq >= noise_freq_range[0]) & (positive_freq <= noise_freq_range[1])

        signal_power = np.sum(positive_magnitude[signal_mask]**2)
        noise_power = np.sum(positive_magnitude[noise_mask]**2)

        # Calculate SNR
        SNR = 10 * np.log10(signal_power / noise_power)
        print(f'Channel {j+1} SNR: {SNR:.2f} dB')

        # Plotting each frequency component in the same figure
        #plt.plot(positive_freq, 20*np.log10(positive_magnitude), label=f'Channel {j+1} - SNR: {SNR:.2f} dB')  # Plot in dB
        plt.plot(positive_freq, 20*np.log10(positive_magnitude) - np.max(20*np.log10(positive_magnitude)), label=f'Channel {j+1}')  # Convert magnitude to dB
    print(np.min(20*np.log10(positive_magnitude))-5)
    plt.xlabel('Frequency (Hz)', fontsize=22)
    plt.ylabel('Magnitude (dB)', fontsize=22)
    plt.xlim(3500, frec_spek)
    plt.ylim(np.min(20*np.log10(positive_magnitude))-65,5)  # Adjust the y-axis limits appropriately
    plt.grid(True)
    plt.legend(loc='best', fontsize='xx-large', frameon=True, shadow=True, borderpad=1)
    plt.show()

"""


def plot_fft_with_zero_padding(data, sample_rate, frec_spek, signal_freq_range, noise_freq_range,Title="Bilde1"):
    """
    Plot the FFT of multiple signals with zero-padding and Hann window applied and calculate SNR.

    Parameters:
    data (numpy.ndarray): The input signals, expected shape (samples, channels).
    sample_rate (float): The sampling rate of the signals.
    frec_spek (float): The maximum frequency to be plotted.
    signal_freq_range (tuple): The frequency range considered as signal (start_freq, end_freq).
    noise_freq_range (tuple): The frequency range considered as noise (start_freq, end_freq).
    """

    plt.figure(figsize=(22, 8))
    

    for j in range(data.shape[1]):  # Iterate over channels
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

        # Calculate power in the signal and noise frequency ranges
        signal_mask = (positive_freq >= signal_freq_range[0]) & (positive_freq <= signal_freq_range[1])
        noise_mask = (positive_freq >= noise_freq_range[0]) & (positive_freq <= noise_freq_range[1])

        signal_power = np.sum(positive_magnitude[signal_mask]**2)
        noise_power = np.sum(positive_magnitude[noise_mask]**2)

        # Calculate SNR
        SNR = 10 * np.log10(signal_power / noise_power)
        print(f'Channel {j+1} SNR: {SNR:.2f} dB')

        # Plotting each frequency component in the same figure
        #plt.plot(positive_freq, 20*np.log10(positive_magnitude), label=f'Channel {j+1} - SNR: {SNR:.2f} dB')  # Plot in dB
        plt.plot(positive_freq, 20*np.log10(positive_magnitude) - np.max(20*np.log10(positive_magnitude)), label=f'Channel {j+1}')  # Convert magnitude to dB
    print(np.min(20*np.log10(positive_magnitude))-5)
    plt.xlabel('Frequency (Hz)', fontsize=22)
    plt.ylabel('Magnitude (dB)', fontsize=22)
    plt.xlim(0, frec_spek)
    plt.ylim(np.min(20*np.log10(positive_magnitude))-100,5)  # Adjust the y-axis limits appropriately
    plt.grid(True)
    plt.title(Title)
    plt.legend(loc='best', fontsize='xx-large', frameon=True, shadow=True, borderpad=1)
    plt.show()












frec_spek = 5
signal_freq_range = (0.9, 1.2)
noise_freq_range = (1.2, 8) 



#sample_rate, data = raspi_import('data_num/test_frekvens')
#plot_fft_with_zero_padding(data, 31250, frec_spek,signal_freq_range,noise_freq_range)


print("Normal puls")

sample_rate, data = raspi_import('data_num/ok1')
plot_data(data)
plot_fft_with_zero_padding(data, 30, frec_spek,signal_freq_range,noise_freq_range,"Ok1")

sample_rate, data = raspi_import('data_num/ok2')
plot_fft_with_zero_padding(data, 30, frec_spek,signal_freq_range,noise_freq_range,"Ok2")

sample_rate, data = raspi_import('data_num/ok4')
plot_fft_with_zero_padding(data, 30, frec_spek,signal_freq_range,noise_freq_range,"Ok4")

sample_rate, data = raspi_import('data_num/ok5')
plot_fft_with_zero_padding(data, 30, frec_spek,signal_freq_range,noise_freq_range, "Ok5")

sample_rate, data = raspi_import('data_num/ok6')
plot_fft_with_zero_padding(data, 30, frec_spek,signal_freq_range,noise_freq_range,"Ok6")


print("Lånt data")

sample_rate, data = raspi_import('data_num/test_lab')
plot_fft_with_zero_padding(data, 30, frec_spek,signal_freq_range,noise_freq_range, "Lånt")


sample_rate, data = raspi_import('data_num/palina1')
plot_fft_with_zero_padding(data, 30, frec_spek,signal_freq_range,noise_freq_range, "reflektans1")

sample_rate, data = raspi_import('data_num/palina2')
plot_fft_with_zero_padding(data, 30, frec_spek,signal_freq_range,noise_freq_range,"reflektans2")


print("robust test")
signal_freq_range = (1.8, 2.2)
noise_freq_range = (1.2, 1.8) 
sample_rate, data = raspi_import('data_num/palina_r1')
plot_fft_with_zero_padding(data, 30, frec_spek,signal_freq_range,noise_freq_range, "robust test 1")

sample_rate, data = raspi_import('data_num/palina_r2')
plot_fft_with_zero_padding(data, 30, frec_spek,signal_freq_range,noise_freq_range, "robust test 2")



sample_rate, data = raspi_import('data_num/test_frekvens')
plot_fft_with_zero_padding(data, 30, frec_spek,signal_freq_range,noise_freq_range, "skjermen med frekvens")

x_1 =data[:, 0]  
x_2 =data[:, 1]  
x_3 =data[:, 2]  

r_21 = np.abs(np.correlate(x_2, x_1, "full"))
r_31 = np.abs(np.correlate(x_3, x_1, "full"))
r_32 = np.abs(np.correlate(x_3, x_2, "full"))


print("r_21: ", r_21)
print("r_31: ", r_31)
print("r_32: ", r_32)

plt.plot(r_21)
plt.plot(r_31)
plt.plot(r_32)
plt.show()


x_1_idx_maxval = np.argmax(r_21)
x_2_idx_maxval = np.argmax(r_31)
x_3_idx_maxval = np.argmax(r_32)

autocorr = np.argmax(np.correlate(x_3, x_3, "full"))
print("Autocorr:\t", autocorr)

n_21 = x_1_idx_maxval - autocorr
n_31 = x_2_idx_maxval - autocorr
n_32 = x_3_idx_maxval - autocorr

print("Delays:\t", n_21, n_31, n_32)

#tau21, tau31, tau32= finn_tidsforsinkelser(signal1, signal2, signal3, 1/30)
#print(tau21, tau31, tau32)
