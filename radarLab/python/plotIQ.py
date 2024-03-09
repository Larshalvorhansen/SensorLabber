import math
# import numpy as np
import matplotlib.pyplot as plt
import csv

x = [] 
y = []
z = []

with open('radarLab/data/filter1.csv','r') as csvfile: 
    hei = csv.reader(csvfile, delimiter = ',')

for row in hei: 
        x.append(row[0]) 
        y.append(int(row[2])) 
        z.append(int(row[3])) 

# vec = []
# vec2 = []

# for i in range(0,100):
#     print(i)
#     vec.append(i)
#     vec2.append(100-i)

# print(vec)
# print(vec2)

plt.plot(x,y,z)
plt.show()