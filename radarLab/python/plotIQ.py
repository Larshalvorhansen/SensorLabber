import math
# import numpy as np
import matplotlib.pyplot as plt
import csv

x = [] 
y = []

with open('radarLab/data/BigBrainHighIQdata.csv','r') as csvfile: 
    plots = csv.reader(csvfile, delimiter = ',')

for row in plt: 
        x.append(row[0]) 
        y.append(int(row[2])) 

vec = []
vec2 = []

for i in range(0,100):
    print(i)
    vec.append(i)
    vec2.append(100-i)


print(vec)
print(vec2)

plt.plot(vec,vec2)
plt.show()