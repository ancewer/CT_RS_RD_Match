import numpy as np
import os, re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def parse_srim_file(file_path, is_carbon=False):
    """
    Parse SRIM output file to extract energy and projected range.

    Parameters:
    - file_path (str): Path to the SRIM output file
    - is_carbon (bool): True if the file is for carbon (MeV/u), False for hydrogen (MeV)

    Returns:
    - energies (list): List of energies in MeV
    - ranges (list): List of projected ranges in mm
    """
    energies = []
    ranges = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        data_section = False

        for line in lines:
            # Skip header until data section starts
            if '--------------' in line:
                data_section = True
                continue
            if '-------------------' in line:
                break  # Stop at unit conversion table

            if data_section and line.strip():
                # Split line into columns (handle multiple spaces)
                columns = re.split(r'\s+', line.strip())
                if len(columns) < 4:
                    continue

                # Extract energy
                energy_str = columns[0] + columns[1]
                if 'GeV' in energy_str and '/' not  in line.strip():
                    energy = float(energy_str.replace('GeV', '')) * 1000  # Convert GeV to MeV
                elif 'MeV' in energy_str and '/' not  in line.strip():
                    energy = float(energy_str.replace('MeV', ''))
                else:
                    continue

                # For carbon, convert MeV/u to MeV (multiply by 12 for carbon-12)
                if is_carbon:
                    energy /= 12

                # Extract projected range (in mm)
                range_str = columns[4]+columns[5]
                if 'mm' in range_str:
                    range_val = float(range_str.replace('mm', ''))
                else:
                    continue  # Skip if range unit is not mm

                energies.append(energy)
                ranges.append(range_val)

    return energies, ranges


def calculate_proton_energy(r80_meas):
    """
    Calculate initial proton energy E_0 (MeV) from measured range r80_meas (mm)
    using the given formula.

    Parameters:
    - r80_meas (array): Measured range in mm

    Returns:
    - E_0 (array): Initial energy in MeV
    """
    log_r = np.log(r80_meas / 10)
    poly = (3.464048 +
            0.561372013 * log_r -
            0.004900892 * log_r ** 2 +
            0.001684756748 * log_r ** 3)
    E_0 = np.exp(poly)
    return E_0

def plot_energy_vs_range(hydrogen_file, carbon_file):
    """
    Plot energy vs projected range for hydrogen and carbon ions.

    Parameters:
    - hydrogen_file (str): Path to hydrogen SRIM file
    - carbon_file (str): Path to carbon SRIM file
    """
    # Parse data from files
    h_energies, h_ranges = parse_srim_file(hydrogen_file, is_carbon=False)
    c_energies, c_ranges = parse_srim_file(carbon_file, is_carbon=True)

    # Create plot
    plt.figure(figsize=(8, 6))

    # Plot hydrogen data
    plt.plot(h_energies, h_ranges, 'b*', label='Hydrogen (H, MeV/u)', marker='o', markersize=5)
    # Generate range values (mm)
    r80_meas = np.linspace(10, 400, 1000)  # 10 mm to 400 mm
    # Calculate energies
    E_0 = calculate_proton_energy(r80_meas)
    # Plot formula-based curve
    plt.plot(E_0, r80_meas, 'b-', label='Proton (Formula)')

    # Plot carbon data
    plt.plot(c_energies, c_ranges, 'r.', label='Carbon (C, MeV/u)', marker='s', markersize=5)
    new_c_energies = fit_and_plot_carbon_formula(c_energies, c_ranges)
    plt.plot(new_c_energies, c_ranges, 'r-', label='Carbon (Formula)')

    # Customize plot
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Projected Range (mm)')
    plt.title('Energy vs Projected Range in H2O (gas, 1 g/cm³)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Use logarithmic scale for energy to better visualize wide range
    # plt.xscale('log')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('energy_vs_range.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_carbon_energy(r80_meas, coeffs):
    """
    Calculate carbon energy E_0 (MeV) from range r80_meas (mm) using fitted coefficients.

    Parameters:
    - r80_meas (array): Measured range in mm
    - coeffs (tuple): Fitted coefficients (a, b, c, d)

    Returns:
    - E_0 (array): Energy in MeV
    """
    log_r = np.log(np.array(r80_meas) / 10)
    a, b, c, d = coeffs
    return np.exp(a + b * log_r + c * log_r ** 2 + d * log_r ** 3)

def carbon_energy_model(log_r, a, b, c, d):
    """
    Model for fitting: ln(E_0) = a + b*log_r + c*log_r^2 + d*log_r^3

    Parameters:
    - log_r (array): ln(r80_meas / 10)
    - a, b, c, d (float): Polynomial coefficients

    Returns:
    - ln(E_0) (array): Natural log of energy in MeV
    """
    return a + b * log_r + c * log_r ** 2 + d * log_r ** 3

def fit_and_plot_carbon_formula(energies, ranges):
    """
    Fit a formula for carbon energy vs range and plot with SRIM data and proton curve.

    Parameters:
    - srim_file (str): Path to carbon SRIM file
    """
    # Compute log_r for fitting
    log_r = np.log(np.array(ranges) / 10)
    ln_E = np.log(energies)

    # Fit the formula
    initial_guess = [3.5, 0.5, 0, 0]  # Initial guess for a, b, c, d
    coeffs, _ = curve_fit(carbon_energy_model, log_r, ln_E, p0=initial_guess)

    # Print the fitted formula
    a, b, c, d = coeffs
    print(f"Fitted formula for carbon:")
    print(f"E_0 = np.exp({a:.6f} + {b:.6f} * log_r + {c:.6f} * log_r**2 + {d:.6f} * log_r**3)")
    print(f"where log_r = np.log(r80_meas / 10), E_0 in MeV, r80_meas in mm")

    # Generate range values for plotting
    r80_meas = ranges

    # Calculate energies
    E_carbon = calculate_carbon_energy(r80_meas, coeffs)

    return E_carbon


# Example usage
if __name__ == "__main__":
    hydrogen_file = "data/Hydrogen50_250MeV  in  H2-O (gas).txt"
    carbon_file = "data/Carbon50-500MeV_u  in  H2-O (gas).txt"
    plot_energy_vs_range(hydrogen_file, carbon_file)