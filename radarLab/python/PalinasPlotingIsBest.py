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
def plot_data(data,filename='plot',xmin = 0,xmax= 1):
    # Number of channels
    num_channels = data.shape[1]
    
    # Time array (assuming equal spacing)
    sample_period = 32e-6  # example value, replace with your actual sample period
    time = np.arange(data.shape[0]) * sample_period

    # Plot each channel in a separate subplot
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

    # Plot all channels in one plot for comparison
    plt.figure(figsize=(22, 8))
    for i in range(num_channels):
        plt.plot(time, data[:, i]+i, label=f'Channel {i+1}')

    plt.xlabel('Time (s)',fontsize=22)
    plt.ylabel('Amplitude [mV]',fontsize=22)
    #plt.title('All Channels',fontsize=16)
    plt.xlim(xmin, xmax)
    plt.grid()
    plt.legend(loc='best', fontsize='xx-large', frameon=True, shadow=True, borderpad=1)
    plt.savefig(f'{filename}_all_channels.png', dpi=300, bbox_inches='tight')


# FFT og ovs ....
    



