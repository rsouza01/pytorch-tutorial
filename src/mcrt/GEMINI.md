# Prompt

Imagine I want to create a simulator to fire photons from some surface in random directions with momenta according to some distribution (blackbody for example) and these photons should go through different layers of stuff (some medium like atmosphere or a thick gas). In the end I want to count the arriving photons and their energies, plot some histograms and so on. I want to use numpy and scipy. Can you generate some code for that? Fully commented please.


# Photon Transport Simulator Code

What you're describing is a Monte Carlo Radiative Transfer (MCRT) simulation. It's a powerful technique used in fields like astrophysics, atmospheric science, and computer graphics.

I will generate a complete, well-commented Python script that accomplishes your goal. This script will:

1. Define the physical and simulation parameters.
2. Generate photons from a flat surface with energies sampled from a blackbody distribution at a given temperature.
3. Propagate these photons through a series of user-defined atmospheric/gas layers.
4. Model absorption and scattering events within the layers.
5. Detect photons that successfully pass through all layers.
6. Analyze and plot the results, showing the initial vs. final energy distributions.

The code uses numpy for efficient array operations, scipy for physical constants, and matplotlib for plotting.

``` bash

pip install numpy scipy matplotlib

```


``` python

#
# Photon Transport Monte Carlo Simulator
#
# This script simulates the journey of photons from a blackbody source
# through a multi-layered medium. It tracks their absorption and
# scattering to determine which photons arrive at a detector.
#
# Dependencies: numpy, scipy, matplotlib
# To install them, run: pip install numpy scipy matplotlib
#

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

# --- 1. SIMULATION CONFIGURATION ---

# Number of photons to simulate. More photons = more accurate results, but slower.
N_PHOTONS = 50000

# Source characteristics
SOURCE_TEMP = 5800  # Temperature of the blackbody source in Kelvin (e.g., 5800K for the Sun)

# Medium properties
# Define the layers of the medium. Each layer is a dictionary with:
# 'thickness': The thickness of the layer in meters.
# 'absorption_coeff': The absorption coefficient in m^-1. This value represents
#                     the probability per unit length of a photon being absorbed.
# 'scattering_coeff': The scattering coefficient in m^-1. This represents
#                     the probability per unit length of a photon being scattered.
#
# The total attenuation is the sum of these two coefficients.
# A higher coefficient means the medium is more opaque.
LAYERS = [
    {
        'name': 'Lower Atmosphere',
        'thickness': 10000,  # 10 km
        'absorption_coeff': 1e-5, # Weak absorption
        'scattering_coeff': 5e-5, # Stronger scattering (like blue sky)
    },
    {
        'name': 'Thick Gas Cloud',
        'thickness': 5000,   # 5 km
        'absorption_coeff': 2e-4, # Stronger absorption
        'scattering_coeff': 1e-4, # Weaker scattering
    },
    {
        'name': 'Upper Atmosphere',
        'thickness': 20000,  # 20 km
        'absorption_coeff': 1e-6, # Very weak absorption
        'scattering_coeff': 1e-5,
    }
]

# --- 2. PHYSICAL CONSTANTS (from Scipy) ---

h = constants.h      # Planck constant (J·s)
c = constants.c      # Speed of light (m/s)
k_B = constants.k    # Boltzmann constant (J/K)
b = constants.Wien   # Wien's displacement law constant (m·K)


# --- 3. HELPER FUNCTIONS ---

def planck_law(wavelength, T):
    """
    Calculates the spectral radiance of a blackbody using Planck's Law.
    This is proportional to the probability of emitting a photon at a given wavelength.

    Args:
        wavelength (float or np.ndarray): Wavelength(s) in meters.
        T (float): Temperature of the blackbody in Kelvin.

    Returns:
        float or np.ndarray: Spectral radiance (W·sr⁻¹·m⁻³).
    """
    # Numerator of Planck's Law
    term1 = 2 * h * c**2 / wavelength**5
    # Denominator term
    term2 = np.expm1(h * c / (wavelength * k_B * T)) # expm1(x) is exp(x) - 1
    # Avoid division by zero for very small wavelengths or zero denominator
    return term1 / term2

def sample_blackbody_wavelength(T):
    """
    Samples a photon wavelength from a blackbody distribution at temperature T.
    This function uses the rejection sampling method.

    Args:
        T (float): Temperature of the blackbody in Kelvin.

    Returns:
        float: A randomly sampled wavelength in meters.
    """
    # Wien's displacement law gives the peak wavelength, which we use to
    # set the scale for our sampling range.
    lambda_max = b / T
    x_range_max = 10 * lambda_max  # Sample over a reasonable range of wavelengths
    y_max = planck_law(lambda_max, T) # Peak value of the Planck distribution

    while True:
        # Generate a random point (x, y) in the bounding box
        # x is a random wavelength, y is a random "probability"
        rand_lambda = np.random.uniform(1e-9, x_range_max) # Start from 1nm to avoid division by zero
        rand_y = np.random.uniform(0, y_max)

        # Rejection criterion: if the random point is under the Planck curve, accept it
        if rand_y <= planck_law(rand_lambda, T):
            return rand_lambda

def get_layer_properties(z_pos):
    """
    Finds the properties of the layer at a given z-position.

    Args:
        z_pos (float): The photon's current position along the z-axis.

    Returns:
        tuple or None: (absorption_coeff, scattering_coeff) for the current layer,
                       or None if the photon is outside the defined layers.
    """
    current_z = 0.0
    for layer in LAYERS:
        if current_z <= z_pos < current_z + layer['thickness']:
            return layer['absorption_coeff'], layer['scattering_coeff']
        current_z += layer['thickness']
    return None, None # Outside all layers


# --- 4. MAIN SIMULATION LOOP ---

def run_simulation():
    """
    Executes the main Monte Carlo simulation.
    """
    # Get the total thickness of all layers to define the detector position
    total_thickness = sum(layer['thickness'] for layer in LAYERS)

    # Lists to store the energy of every photon
    initial_energies = []
    final_energies = []

    print(f"Starting simulation with {N_PHOTONS} photons...")

    for i in range(N_PHOTONS):
        # --- a. Photon Initialization ---

        # Generate a photon with energy from the blackbody distribution
        wavelength = sample_blackbody_wavelength(SOURCE_TEMP)
        energy = h * c / wavelength
        initial_energies.append(energy)

        # Initial position is the origin (z=0)
        pos = np.array([0.0, 0.0, 0.0])

        # Initial direction is random in the forward hemisphere (+z)
        # To get an isotropic distribution, we sample cos(theta) uniformly
        cos_theta = np.sqrt(np.random.uniform(0, 1)) # sqrt for Lambertian emission pattern
        theta = np.arccos(cos_theta)
        phi = np.random.uniform(0, 2 * np.pi)

        # Convert spherical coordinates to a Cartesian direction vector
        direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)  # Always positive, so photon starts moving forward
        ])

        # --- b. Photon Propagation ---
        is_absorbed = False
        while True:
            # Check if photon has exited the simulation domain
            if pos[2] >= total_thickness:
                # Photon reached the detector
                final_energies.append(energy)
                break
            if pos[2] < 0:
                # Photon scattered backwards out of the simulation
                break

            # Get properties of the current layer
            abs_coeff, scat_coeff = get_layer_properties(pos[2])

            if abs_coeff is None or (abs_coeff == 0 and scat_coeff == 0):
                # Photon is in a vacuum or outside layers, propagate to next boundary
                # For simplicity, we assume this means it escapes if it's beyond the last layer.
                # This case is largely handled by the z >= total_thickness check.
                break

            # Total attenuation coefficient
            att_coeff = abs_coeff + scat_coeff

            # Sample the distance to the next interaction event from an exponential distribution
            # This is the "mean free path" logic.
            distance_to_interaction = -np.log(np.random.rand()) / att_coeff

            # Move the photon to the interaction point
            pos += distance_to_interaction * direction

            # Determine if the photon was absorbed or scattered
            # The probability of scattering is the ratio of scattering to total attenuation
            if np.random.rand() < (scat_coeff / att_coeff):
                # --- SCATTERING ---
                # The photon changes direction isotropically.
                # For simplicity, we assume elastic scattering (no energy change).
                # Generate a new random direction vector on a unit sphere.
                u = np.random.uniform(-1, 1)
                theta = np.arccos(u)
                phi = np.random.uniform(0, 2 * np.pi)
                direction = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
            else:
                # --- ABSORPTION ---
                is_absorbed = True
                break # Photon's journey ends here

        if i % (N_PHOTONS // 10) == 0 and i > 0:
            print(f"  ... simulated {i}/{N_PHOTONS} photons")

    print("Simulation finished.")
    return np.array(initial_energies), np.array(final_energies)


# --- 5. ANALYSIS AND PLOTTING ---
if __name__ == "__main__":
    initial_energies, final_energies = run_simulation()

    # --- Print Summary Statistics ---
    num_started = len(initial_energies)
    num_arrived = len(final_energies)
    transmission = (num_arrived / num_started) * 100 if num_started > 0 else 0

    print("\n--- Simulation Results ---")
    print(f"Photons Started: {num_started}")
    print(f"Photons Arrived at Detector: {num_arrived}")
    print(f"Overall Transmission: {transmission:.2f}%")

    if num_arrived > 0:
        avg_initial_energy = np.mean(initial_energies)
        avg_final_energy = np.mean(final_energies)
        print(f"Average Initial Energy (eV): {avg_initial_energy / constants.e:.2f}")
        print(f"Average Energy of Arrived Photons (eV): {avg_final_energy / constants.e:.2f}")

    # --- Create Histograms ---
    # Convert energies from Joules to electron-Volts (eV) for more intuitive numbers
    initial_eV = initial_energies / constants.e
    final_eV = final_energies / constants.e

    plt.style.use('seaborn-v0_8-talk')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define bins for the histogram. Let's base them on the initial distribution.
    bins = np.linspace(0, np.percentile(initial_eV, 99.9), 100)

    # Plot initial energy distribution (all photons created)
    ax.hist(initial_eV, bins=bins, color='royalblue', alpha=0.7, label='Initial Photon Energies (Source)')

    # Plot final energy distribution (only photons that arrived)
    if num_arrived > 0:
        ax.hist(final_eV, bins=bins, color='darkorange', alpha=0.8, label=f'Arriving Photon Energies ({num_arrived} detected)')

    ax.set_title(f"Photon Energy Distribution (Source Temp: {SOURCE_TEMP}K)")
    ax.set_xlabel("Photon Energy (eV)")
    ax.set_ylabel("Number of Photons")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

```

# Interpreting the Output

When you run the script, you will see:

1. Console Output: The progress of the simulation will be printed, followed by a summary of the results, including the percentage of photons that successfully passed through the medium.
2. A Plot Window: A histogram will appear.
    * The blue histogram shows the energy distribution of all photons as they were created at the source. This is the characteristic blackbody spectrum for the temperature you set.
    * The orange histogram shows the energy distribution of only the photons that made it to the detector.

You will likely observe that the orange histogram is much smaller than the blue one, representing the photons that were absorbed or scattered away. You might also notice a change in the shape of the distribution if your absorption coefficients were dependent on energy (a feature you could add for a more advanced simulation). In this simplified model where absorption is constant with energy, the shape of the final distribution will largely mirror the initial one, just scaled down.


# Next steps

Take your time to go through it. The comments are there to guide you through the logic, from the physics of the blackbody source to the Monte Carlo method for propagation.

Feel free to ask any questions that come up. For instance, you might want to consider:

* How to make the absorption coefficients energy-dependent (which is more realistic for most materials).
* How to implement a different scattering phase function (e.g., forward-scattering instead of isotropic).
* How to visualize the actual 3D paths of a few photons.

