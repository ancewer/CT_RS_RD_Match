# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_config_file(work_dir, template_fie, energy, particles):
    lines = open(template_fie, encoding='utf-8').read().splitlines()
    for i in range(len(lines)):
        if 'd:So/PrimBeam/BeamEnergy' in lines[i]:
            lines[i] = f'd:So/PrimBeam/BeamEnergy = {energy * 12} MeV'
        elif 'i:So/PrimBeam/NumberOfHistoriesInRun' in lines[i]:
            lines[i] = f'i:So/PrimBeam/NumberOfHistoriesInRun = {particles}'
        elif 's:Sc/IDD/OutputFile' in lines[i]:
            lines[i] = f's:Sc/IDD/OutputFile = "{energy}MeV_idd_result"'

    new_fname = os.path.join(work_dir, f'{energy}MeV_carbon_idd.txt')
    with open(new_fname, 'w', newline='\n') as fid:
        for line in lines:
            fid.write(f'{line}\n')
    return new_fname


def run_topas_simulation(WorkDir, fname):
    cur_dir = os.getcwd()
    os.chdir(WorkDir)
    with open(os.path.join(WorkDir, 'run_topas.sh'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(f"#!/bin/bash\nset -x\nsource ~/.bashrc\nexport TOPAS_G4_DATA_DIR=~/topas/G4Data\n"
                f"export PATH=$PATH:~/topas/bin\ntopas {fname}")
    os.system("wsl bash -c 'bash \\run_topas.sh'")
    os.chdir(cur_dir)


def calculate_r80(depth, dose):
    """
    Calculate R80: the depth where dose falls to 80% of the maximum on the distal side.

    Parameters:
    - depth (array): Depth values in cm.
    - dose (array): Dose values in Gy.

    Returns:
    - r80 (float): Depth (cm) at R80.
    - dose_r80 (float): Dose (Gy) at R80.
    """
    # Find the maximum dose and its depth (Bragg peak)
    max_dose_idx = np.argmax(dose)
    max_dose = dose[max_dose_idx]
    max_depth = depth[max_dose_idx]

    # Target dose is 80% of maximum
    target_dose = 0.8 * max_dose

    # Search on the distal side (depths > max_depth)
    distal_indices = np.where(depth > max_depth)[0]
    for i in distal_indices:
        if dose[i] <= target_dose:
            # Interpolate between this point and the previous point
            x0, x1 = depth[i - 1], depth[i]
            y0, y1 = dose[i - 1], dose[i]
            # Linear interpolation: r80 = x0 + (x1 - x0) * (target_dose - y0) / (y1 - y0)
            r80 = x0 + (x1 - x0) * (target_dose - y0) / (y1 - y0)
            dose_r80 = target_dose
            return r80, dose_r80

    raise ValueError("R80 not found within the data range.")

def plot_idd(input_csv_path, output_plot_path, energy="50"):
    """
    Plot Integral Depth Dose (IDD) from a TOPAS CSV file and save the plot.

    Parameters:
    - input_csv_path (str): Path to the TOPAS CSV file (e.g., '50.0_idd_result.csv').
    - output_plot_path (str): Path to save the plot (e.g., 'idd_50MeV_carbon.png').
    - title (str): Plot title (default: 'Integral Depth Dose (IDD)').
    - xlim (tuple): X-axis limits (default: (0, 5) to focus on Bragg peak).
    - yscale (str): Y-axis scale ('linear' or 'log', default: 'linear').
    - particle (str): Particle type for label (default: 'Carbon Ion').
    - energy (str): Beam energy for label (default: '50 MeV').

    Returns:
    - fig: Matplotlib figure object for further customization.

    Raises:
    - FileNotFoundError: If input_csv_path does not exist.
    - ValueError: If yscale is invalid.
    """
    title = "Integral Depth Dose (IDD)"
    particle = "Carbon Ion"
    # Validate input file
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"The file {input_csv_path} does not exist.")

    # Read CSV, skipping header lines (lines starting with '#')
    data = pd.read_csv(input_csv_path, comment='#', header=None,
                       names=['R', 'Phi', 'Z', 'Dose'])

    # Calculate depth in cm (Z * 0.1 cm, as each bin is 0.1 cm)
    data['Depth'] = data['Z'] * 0.1

    # Extract depth and dose
    depth = data['Depth']
    dose = data['Dose']

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(depth, dose, label=f"{energy}MeV {particle}", color='blue')
    ax.set_xlabel('Depth in Water (cm)')
    ax.set_ylabel('Dose (Gy)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    # Calculate R80 and dose at R80
    r80, dose_r80 = calculate_r80(depth, dose)
    print(f"R80: {r80:.3f} cm, Dose at R80: {dose_r80:.2e} Gy")

    # Set plot limits
    xlim = (0, r80 + 5)  # Extend x-axis to R80 + 5 cm
    ax.set_xlim(xlim)
    # Ensure output directory exists
    output_dir = os.path.dirname(output_plot_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the plot
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_plot_path}")
    # plt.show()

work_dir = os.path.join(os.getcwd(), 'work_dir')
os.makedirs(work_dir, exist_ok=True)
template_fie = os.path.join('', 'carbon_idd_template.txt')
energies = np.arange(50, 500, 0.5)
particles = 500000
for energy in energies:
    fname = create_config_file(work_dir, template_fie, energy, particles)
    run_topas_simulation(work_dir, os.path.basename(fname))
    plot_idd(os.path.join(work_dir, f'{energy}MeV_idd_result.csv'), work_dir, energy=str(energy))

