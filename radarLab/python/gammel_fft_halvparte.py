
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


def calculate_fft_with_zero_padding(data, sample_rate,frec_spek,signal_freq_range, noise_freq_range):
    """
    Calculate the FFT of each channel in the data with zero-padding and apply a Hann window.

    Parameters:
    - data (numpy.ndarray): Input signal data, shape (samples, channels).
    - sample_rate (float): Sampling rate of the data.

    Returns:
    - A tuple containing:
      - full_freq (numpy.ndarray): Frequency bins array.
      - full_magnitude (list of numpy.ndarray): Magnitudes of FFT for each channel.
      - SNR_values (list): SNR values for each channel.
      - peak_frequency (value/list): Peak frequency and value.
    """
    channels = data.shape[1]
    full_magnitude = []
    SNR_values = []
    peak_frequencies = []
    peak_magnitudes = []

    for j in range(channels):
        d = data[:, j] - np.mean(data[:, j])  # Detrend the signal
        windowed_data = d * np.hanning(len(d))  # Apply Hann window

        # Zero-padding
        N = len(windowed_data)
        N_padded = 2**np.ceil(np.log2(N)).astype(int)

        # FFT
        fft_result = fft(windowed_data, n=N_padded)
        magnitude = np.abs(fft_result)

        # Store magnitudes and calculate SNR for each channel
        full_magnitude.append(magnitude)
        freq = fftfreq(N_padded, 1 / sample_rate)
        SNR_values.append(calculate_SNR(freq[:N_padded//2], signal_freq_range, noise_freq_range, magnitude[:N_padded//2]))
       
        # freq and magnitude are for the current channel
        peak_frequency, peak_magnitude = find_peak_frequency(-frec_spek, frec_spek, freq[:N_padded//2], full_magnitude[j][:N_padded//2])
        peak_frequencies.append(peak_frequency)
        peak_magnitudes.append(peak_magnitude)

    return freq, full_magnitude, SNR_values, peak_frequency, peak_magnitude


def plot_fft_results(freq, magnitudes, SNR_values, frec_spek, title="Bilde1"):
    """
    Plot the FFT results for each channel.

    Parameters:
    - freq (numpy.ndarray): Frequency bins array.
    - magnitudes (list of numpy.ndarray): Magnitudes of FFT for each channel.
    - SNR_values (list): SNR values for each channel.
    - frec_spek (float): Frequency spectrum limit for plotting.
    - title (str): Title of the plot.
    """
    plt.figure(figsize=(22, 8))
    
    for i, (magnitude, SNR_value) in enumerate(zip(magnitudes, SNR_values)):
        plt.plot(freq, 20*np.log10(magnitude) - np.max(20*np.log10(magnitude)), label=f'Channel {i+1} - SNR: {SNR_value:.2f} dB')

    plt.xlabel('Frequency (Hz)', fontsize=22)
    plt.ylabel('Magnitude (dB)', fontsize=22)
    plt.xlim(-frec_spek, frec_spek)
    plt.ylim(-100, 5)  # Example y-axis limits, adjust as needed
    plt.grid(True)
    plt.title(title)
    plt.legend(loc='best', fontsize='xx-large', frameon=True, shadow=True, borderpad=1)
    plt.show()






#funker
    

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
        toppen = find_peak_frequency(-frec_spek, frec_spek, full_freq, full_magnitude)
    
    
    
    
    plt.xlabel('Frequency (Hz)', fontsize=22)
    plt.ylabel('Magnitude (dB)', fontsize=22)
    plt.xlim(-frec_spek, frec_spek)
    plt.ylim(np.min(20*np.log10(full_magnitude))-100,5)  # Adjust y-axis limits appropriately
    plt.grid(True)
    plt.title(Title)
    plt.legend(loc='best', fontsize='xx-large', frameon=True, shadow=True, borderpad=1)
    plt.show()
    