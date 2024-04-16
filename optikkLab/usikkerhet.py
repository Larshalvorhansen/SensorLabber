
import sys
import numpy as np #Plotting
import matplotlib.pyplot as plt #Lese CSV
import csv
from scipy.fft import fft, ifft,fftfreq
from math import atan, sqrt
# Plotting 
plt.rc('xtick', labelsize=19) # endre størrelsen på x-tall
plt.rc('ytick', labelsize=19) # endre størrelse på y-tall
plt.rcParams["figure.figsize"] = [8, 6] # endre størrelse på alle bildene 
#plt.rcParams["figure.autolayout"] = True
plt.rcParams['lines.linewidth'] = 2.5


# Det forskjellige normalpulsene 
normal_grønn = np.array([70,71,69,73,71,70,68,71,68,72,69])
print(normal_grønn*60)

normal_blå = np.array([1, 1.057, 1.019, 0.977, 1.048])

#vinkler_min_31 = np.array([31.57, 23.41, 23.41, 23.41, 22.77])


# Beregn standardavviket for disse vinklene
standardavvik_grønn = np.std(normal_grønn)
standardavvik_blå = np.mean(normal_grønn)
#standardavvik_blå = np.std(normal_blå)
#standardavvik_31 = np.std(vinkler_min_31)

# Beregn variansen for disse vinklene


print(f"Standardavvik blå: {standardavvik_grønn}")


print(f"Standardavvik grønn: {standardavvik_blå}")


