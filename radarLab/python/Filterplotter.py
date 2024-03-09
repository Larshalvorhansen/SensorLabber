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



#Eksportere data
header = []
data = []
filename = '/Users/polinaermakova/Documents/GitHub/SensorLabber/radarLab/data/filter1.csv'
#Henter data fra csvfil
with open(filename) as csvfile:
    csvreader = csv.reader(csvfile)
    #Leser første linje i csv-fila (den med navn til kanalene)
    header = next(csvreader)
    for datapoint in csvreader:
        values = [float(value) for value in datapoint]
        data.append(values)
#Legger inn data fra hver kanal i hver sin liste
time1 = [(p[0]) for p in data]
data4=[(p[1]) for p in data] #inngangsignalet 
data5 = [(p[2]) for p in data]




fig, (ax) = plt.subplots(1,1, figsize=(22,8))
#Plotsemilogx
ax.plot(time1,data4, label = 'Inngansignal s(t) ')
ax.plot(time1,data5, label = 'Utgangsignal y(t) ')
#ax.plot(time,ch2, label = 'Utgangsignal y(t)')
#Tittel og aksenavn
#ax.set_title('Oscilloskop')
ax.set_xlabel('Frekvens Hz',fontsize=22)
ax.set_ylabel('Demping (dB))',fontsize=22)
#Legend, loc = plassering
ax.legend(loc='upper right') #Rutenett
ax.grid(True)
ax.set_xlabel('Frekvens[Hz]',fontsize=22)
ax.set_ylabel('Dempling [dB]',fontsize=22)



ax.axhline(y=-3, color='r', linestyle='--', label ='Dempning -3dB',)
#ax.axhline(x=29, color='g', linestyle='--', label ='Knekkfrekvens')
ax.axvline(x=43, color='g', linestyle=':', label ='Knekfrekvense f$_c$ = 43 Hz')
#Legend, loc = plassering
ax.legend(loc='upper right') #Rutenett
ax.grid(True)
ax.set_ylim(-30,10)
#ax.set_xlim(0, 50)
ax.legend(loc='best',fontsize=20) #Rutenett
ax.grid(True)
plt.show()

#fig.savefig('signalrespons', dpi=300, bbox_inches='tight')