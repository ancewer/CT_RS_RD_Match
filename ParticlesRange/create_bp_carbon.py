# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.getcwd(), 'work_dir', 'simulation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_config_file(work_dir, template_file, energy, particles):
    """
    Create a TOPAS configuration file from a template with specified energy and particle count.
    """
    logger.info(f"Creating config file for energy: {energy} MeV, particles: {particles}")
    try:
        lines = open(template_file, encoding='utf-8').read().splitlines()
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
        logger.info(f"Config file created: {new_fname}")
        return new_fname
    except Exception as e:
        logger.error(f"Error creating config file for energy {energy} MeV: {str(e)}")
        raise


def run_topas_simulation(work_dir, fname):
    """
    Run TOPAS simulation using the specified configuration file and log execution time.
    """
    logger.info(f"Running TOPAS simulation for config: {fname}")
    try:
        start_time = time.time()
        logger.debug(f"Simulation started at: {time.ctime(start_time)}")

        cur_dir = os.getcwd()
        os.chdir(work_dir)
        script_path = os.path.join(work_dir, 'run_topas.sh')
        with open(script_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(f"#!/bin/bash\nset -x\nsource ~/.bashrc\nexport TOPAS_G4_DATA_DIR=~/topas/G4Data\n"
                    f"export PATH=$PATH:~/topas/bin\ntopas {fname}")
        logger.debug(f"Created run script: {script_path}")
        os.system("wsl bash -c 'bash \\run_topas.sh'")

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Simulation completed for {fname}")
        logger.info(f"Simulation ended at: {time.ctime(end_time)}")
        logger.info(f"Simulation duration: {duration:.2f} seconds")

        os.chdir(cur_dir)
    except Exception as e:
        logger.error(f"Error running simulation for {fname}: {str(e)}")
        os.chdir(cur_dir)
        raise


def calculate_r80(depth, dose):
    """
    Calculate R80: the depth where dose falls to 80% of the maximum on the distal side.
    """
    logger.debug("Calculating R80")
    try:
        max_dose_idx = np.argmax(dose)
        max_dose = dose[max_dose_idx]
        max_depth = depth[max_dose_idx]
        target_dose = 0.8 * max_dose
        distal_indices = np.where(depth > max_depth)[0]
        for i in distal_indices:
            if dose[i] <= target_dose:
                x0, x1 = depth[i - 1], depth[i]
                y0, y1 = dose[i - 1], dose[i]
                r80 = x0 + (x1 - x0) * (target_dose - y0) / (y1 - y0)
                dose_r80 = target_dose
                logger.debug(f"R80 calculated: {r80:.3f} cm, Dose: {dose_r80:.2e} Gy")
                return r80, dose_r80
        logger.error("R80 not found within the data range")
        raise ValueError("R80 not found within the data range.")
    except Exception as e:
        logger.error(f"Error calculating R80: {str(e)}")
        raise


def plot_idd(input_csv_path, output_plot_path, energy="50"):
    """
    Plot Integral Depth Dose (IDD) from a TOPAS CSV file and save the plot.
    """
    logger.info(f"Plotting IDD for {input_csv_path}")
    title = "Integral Depth Dose (IDD)"
    particle = "Carbon Ion"
    try:
        if not os.path.exists(input_csv_path):
            logger.error(f"Input CSV file not found: {input_csv_path}")
            raise FileNotFoundError(f"The file {input_csv_path} does not exist.")

        data = pd.read_csv(input_csv_path, comment='#', header=None,
                           names=['R', 'Phi', 'Z', 'Dose'])
        data['Depth'] = data['Z'] * 0.1
        depth = data['Depth']
        dose = data['Dose']

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(depth, dose, label=f"{energy}MeV {particle}", color='blue')
        ax.set_xlabel('Depth in Water (cm)')
        ax.set_ylabel('Dose (Gy)')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

        r80, dose_r80 = calculate_r80(depth, dose)
        logger.info(f"R80: {r80:.3f} cm, Dose at R80: {dose_r80:.2e} Gy")

        xlim = (0, r80 + 5)
        ax.set_xlim(xlim)

        output_dir = os.path.dirname(output_plot_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.debug(f"Created output directory: {output_dir}")

        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output_plot_path}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error plotting IDD for {input_csv_path}: {str(e)}")
        raise


import datetime


def main():
    """
    Main function to run the simulation for a range of energies with dynamic time estimation.
    """
    logger.info("Starting simulation")
    try:
        work_dir = os.path.join(os.getcwd(), 'work_dir')
        os.makedirs(work_dir, exist_ok=True)
        logger.debug(f"Work directory: {work_dir}")
        template_file = os.path.join('', 'carbon_idd_template.txt')
        energies = np.arange(50, 501, 5)
        particles = 500000
        logger.info(f"Simulation parameters - Num of Energies: {len(energies)}, Particles: {particles}")

        # Track simulation times
        sim_times = []
        initial_sim_count = 3  # Number of initial simulations for first estimate
        update_interval = 10  # Update estimate every 10 simulations
        start_time = datetime.datetime.now()

        def log_estimated_completion(avg_time, remaining_energies, current_time):
            """Log estimated completion time based on average simulation time."""
            total_remaining_time = len(remaining_energies) * avg_time
            estimated_end_time = current_time + datetime.timedelta(seconds=total_remaining_time)
            logger.info(f"Estimated completion time: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')} "
                        f"(avg_time_per_energy: {avg_time:.2f} s, {len(remaining_energies)} energies left)")

        for i, energy in enumerate(energies):
            logger.info(f"Processing energy: {energy} MeV")
            sim_start_time = time.time()

            # Run simulation steps
            fname = create_config_file(work_dir, template_file, energy, particles)
            run_topas_simulation(work_dir, os.path.basename(fname))
            plot_idd(
                os.path.join(work_dir, f'{energy}MeV_idd_result.csv'),
                os.path.join(work_dir, f'{energy}MeV_idd_result.png'),
                energy=str(energy)
            )

            # Record simulation time
            sim_duration = time.time() - sim_start_time
            sim_times.append(sim_duration)
            logger.debug(f"Energy {energy} MeV took {sim_duration:.2f} seconds")

            # Initial estimate after first few simulations
            if i + 1 == initial_sim_count:
                avg_time_per_energy = sum(sim_times) / len(sim_times)
                logger.info(f"Initial avg_time_per_energy after {initial_sim_count} simulations: "
                            f"{avg_time_per_energy:.2f} seconds")
                log_estimated_completion(avg_time_per_energy, energies[i + 1:], datetime.datetime.now())

            # Periodic update of estimate
            if (i + 1) % update_interval == 0 and i + 1 > initial_sim_count:
                avg_time_per_energy = sum(sim_times) / len(sim_times)
                logger.info(f"Updated avg_time_per_energy after {i + 1} simulations: "
                            f"{avg_time_per_energy:.2f} seconds")
                log_estimated_completion(avg_time_per_energy, energies[i + 1:], datetime.datetime.now())

        # Log final completion time
        end_time = datetime.datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        avg_time_per_energy = total_duration / len(energies) if sim_times else 0
        logger.info(f"Simulation completed successfully at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {total_duration:.2f} seconds ({total_duration / 3600:.2f} hours)")
        logger.info(f"Final avg_time_per_energy: {avg_time_per_energy:.2f} seconds")

    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()

