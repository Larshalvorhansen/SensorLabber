import numpy as np


muabo = np.genfromtxt("./optikkLab/muabo.txt", delimiter=",")
muabd = np.genfromtxt("./optikkLab/muabd.txt", delimiter=",")

red_wavelength = 600 # Replace with wavelength in nanometres
green_wavelength = 515 # Replace with wavelength in nanometres
blue_wavelength = 460 # Replace with wavelength in nanometres

wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])

bvf = 0.01 # Blood volume fraction, average blood amount in tissue
oxy = 0.8 # Blood oxygenation

# Absorption coefficient ($\mu_a$ in lab text)
# Units: 1/m
mua_other = 25 # Background absorption due to collagen, et cetera
mua_blood = (mua_blood_oxy(wavelength)*oxy # Absorption due to
            + mua_blood_deoxy(wavelength)*(1-oxy)) # pure blood
mua = mua_blood*bvf + mua_other

# reduced scattering coefficient ($\mu_s^\prime$ in lab text)
# the numerical constants are thanks to N. Bashkatov, E. A. Genina and
# V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
# tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
# Units: 1/m
musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)

# mua and musr are now available as shape (3,) arrays
# Red, green and blue correspond to indexes 0, 1 and 2, respectively

# TODO calculate penetration depth
print(musr,"Musr" )
print(mua, "Mua")

def calculate_penetration_depth(mua, musr):
    return 1 / np.sqrt(3 * (musr + mua) * mua)


# Function to calculate the constant C based on mu_a and mu_s'
def calculate_C(mua, musr):
    return np.sqrt(3 * mua * (musr + mua))

# Function to calculate fluence rate at the surface phi(0) based on penetration depth delta and mu_a
def calculate_phi_0(mua, penetration_depth):
    return 1 / (2 * penetration_depth * mua)

# Function to calculate transmission T based on depth d, mu_a, and mu_s'
def calculate_transmission(d, mua, musr):
    C = calculate_C(mua, musr)
    phi_0 = calculate_phi_0(mua, calculate_penetration_depth(mua, musr))
    phi_d = phi_0 * np.exp(-C * d)
    T = phi_d / phi_0
    return T


print ("Oppgave 1a) ")
# Call the function with the calculated mua and musr values
penetration_depth = calculate_penetration_depth(mua, musr)
print(penetration_depth, "Penetration Depth")


print ("Oppgave 1b) i meter ")
# For demonstration, let's calculate the transmission at a depth of 1 mm for red, green, and blue wavelengths
depth = 1e-3 #1cm  # Depth in meters
transmission_red = calculate_transmission(depth, mua[0], musr[0])
transmission_green = calculate_transmission(depth, mua[1], musr[1])
transmission_blue = calculate_transmission(depth, mua[2], musr[2])

print(transmission_red, transmission_green, transmission_blue)


