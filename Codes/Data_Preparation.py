import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ===============================================================================
# Function to prepare the Simulation data and combine them in a single Data Frame
# ===============================================================================

def process_simulations(base_dir, output_path, sim_start, sim_end, print_option, time_option, plot_sim_num=None):
    """
    Process simulation data, optionally export each simulation to CSV, plot data for a specified simulation, and return a list of DataFrames.

    Parameters:
    base_dir (str): Base directory where the simulation folders are located.
    output_path (str): Path where each simulation's CSV file will be saved (if print_option is 'print').
    sim_start (int): Simulation number start point.
    sim_end (int): Simulation number end point.
    print_option (str): If 'print', export each simulation's data to CSV; if 'nprint', do not.
    time_option (str): If 'deltime', delete the time column; if 'time', keep it.
    plot_sim_num (int, optional): Simulation number to plot.

    Returns:
    List of pd.DataFrame: Each DataFrame contains the data from one simulation.
    """
    param_file_path = os.path.join(base_dir, 'Parameter_Simulation_01.csv')
    param_df = pd.read_csv(param_file_path, delimiter=';', index_col='No.')
    parameter_names = [
        "BendDieBendingAngle",
        "BendDieLateralMovement",
        "ColletAxialForce",
        "ColletAxialMovement",
        "MandrelAxialForce",
        "MandrelAxialMovement",
        "PressureDieAxialForce",
        "PressureDieAxialMovement",
        "PressureDieLateralForce",
        "WiperDieAxialForce",
        "WiperDieLateralForce",
        "WiperDieLateralMovement"
    ]

    simulations_data = []  # List to store DataFrames for each simulation

    for sim_num in range(sim_start, sim_end + 1):
        folder_name = f"SIM_V12-{str(sim_num).zfill(2)}"
        sim_path = os.path.join(base_dir, folder_name)
        file_names = [
            f"{folder_name}_bend-die_bending-angle_rad.csv",
            f"{folder_name}_bend-die_lateral-movement.csv",
            f"{folder_name}_collet_axial-force.csv",
            f"{folder_name}_collet_axial-movement.csv",
            f"{folder_name}_mandrel_axial-force.csv",
            f"{folder_name}_mandrel_axial-movement.csv",
            f"{folder_name}_pressure-die_axial-force.csv",
            f"{folder_name}_pressure-die_axial-movement.csv",
            f"{folder_name}_pressure-die_lateral-force.csv",
            f"{folder_name}_wiper-die_axial-force.csv",
            f"{folder_name}_wiper-die_lateral-force.csv",
            f"{folder_name}_wiper-die_lateral-movement.csv"
        ]
        temp_data = pd.DataFrame()

        for i, file_name in enumerate(file_names):
            file_path = os.path.join(sim_path, file_name)
            temp_df = pd.read_csv(file_path, delimiter=';', header=None, names=['Time', parameter_names[i]], skiprows=1)
            if parameter_names[i] == "BendDieBendingAngle":
                temp_df[parameter_names[i]] = np.rad2deg(temp_df[parameter_names[i]])
            if i == 0:
                temp_data['Time'] = temp_df['Time']
            temp_data[parameter_names[i]] = temp_df[parameter_names[i]]

        for param in param_df.columns:
            temp_data[param] = param_df.loc[sim_num, param]
        
        if time_option == 'deltime':
            temp_data.drop('Time', axis=1, inplace=True)  # Ensure the 'Time' column is removed here

        simulations_data.append(temp_data)  # Append the simulation DataFrame to the list

        if print_option == 'print':
            output_file_path = os.path.join(output_path, f'Simulation_{sim_num}.csv')
            temp_data.to_csv(output_file_path, index=False)

    if plot_sim_num is not None:
        # Adjust plot_data to work with the updated simulations_data list
        plot_data(simulations_data[plot_sim_num - sim_start], parameter_names)


    return simulations_data

def plot_data(sim_data, parameter_names):
    """
    Plot data for a specified DataFrame.

    Parameters:
    sim_data (pd.DataFrame): DataFrame containing data for one simulation.
    parameter_names (list): List of parameter names to plot.
    """
    # Categorize parameters
    force_params = [param for param in parameter_names if 'Force' in param]
    movement_params = [param for param in parameter_names if 'Movement' in param]
    bending_params = [param for param in parameter_names if 'BendingAngle' in param]

    # Plotting
    plt.figure(figsize=(15, 10))

    # Plot force data
    plt.subplot(3, 1, 1)
    for param in force_params:
        plt.plot(sim_data['Time'], sim_data[param], label=param)
    plt.ylabel('Force [kN]')
    plt.legend()

    # Plot movement data
    plt.subplot(3, 1, 2)
    for param in movement_params:
        plt.plot(sim_data['Time'], sim_data[param], label=param)
    plt.ylabel('Movement [mm]')
    plt.legend()

    # Plot bending angle
    plt.subplot(3, 1, 3)
    for param in bending_params:
        plt.plot(sim_data['Time'], sim_data[param], label=param)
    plt.xlabel('Time')
    plt.ylabel('Angle [degree]')
    plt.legend()

    plt.tight_layout()
    plt.show()


# =============================================
# Function to load geometry data in data frames
# =============================================
def load_and_plot_geometry(base_dir, sim_start, sim_end, plot=False, plot_sims=[]):
    """
    Loads geometry data from simulation folders and optionally plots both before and after springback data,
    including Secondary Axis, Main Axis, Collapse, and Out-of-roundness.

    Parameters:
    - base_dir (str): Base directory where the simulation folders are located.
    - sim_start (int): Starting simulation number.
    - sim_end (int): Ending simulation number.
    - plot (bool): If True, plot the data for specified simulations.
    - plot_sims (list): List of simulation numbers to plot.

    Returns:
    - Tuple of lists: (geometry_before_dataframes, geometry_after_dataframes)
    """
    geometry_before_dataframes = []
    geometry_after_dataframes = []

    for sim_num in range(sim_start, sim_end + 1):
        folder_name = f"SIM_V12-{str(sim_num).zfill(2)}"
        sim_path = os.path.join(base_dir, folder_name)

        file_path_before = os.path.join(sim_path, f"{folder_name}_geometry_before_springback.csv")
        file_path_after = os.path.join(sim_path, f"{folder_name}_geometry_after_springback.csv")

        df_before = pd.read_csv(file_path_before, delimiter=';')
        df_after = pd.read_csv(file_path_after, delimiter=';')

        geometry_before_dataframes.append(df_before)
        geometry_after_dataframes.append(df_after)

        if plot and sim_num in plot_sims:
            plt.figure(figsize=(12, 8))
            # Secondary and Main Axis
            plt.subplot(2, 1, 1)
            plt.plot(df_before['Angle[degree]'], df_before['Secondary axis[mm]'], label='Before - Secondary Axis', linestyle='--')
            plt.plot(df_before['Angle[degree]'], df_before['Main axis[mm]'], label='Before - Main Axis', linestyle='--')
            plt.plot(df_after['Angle[degree]'], df_after['Secondary axis[mm]'], label='After - Secondary Axis')
            plt.plot(df_after['Angle[degree]'], df_after['Main axis[mm]'], label='After - Main Axis')
            plt.title(f"Comparison of Geometry Before and After Springback for Simulation {sim_num}")
            plt.xlabel('Angle (degree)')
            plt.ylabel('Axis Measurements (mm)')
            plt.legend()
            plt.grid(True)

            # Collapse and Out-of-roundness
            plt.subplot(2, 1, 2)
            plt.plot(df_before['Angle[degree]'], df_before['Out-of-roundness[-]'], label='Before - Out-of-roundness', linestyle='--')
            plt.plot(df_before['Angle[degree]'], df_before['Collapse[mm]'], label='Before - Collapse', linestyle='--')
            plt.plot(df_after['Angle[degree]'], df_after['Out-of-roundness[-]'], label='After - Out-of-roundness')
            plt.plot(df_after['Angle[degree]'], df_after['Collapse[mm]'], label='After - Collapse')
            plt.xlabel('Angle (degree)')
            plt.ylabel('Out-of-roundness and Collapse')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return (geometry_before_dataframes, geometry_after_dataframes)


# =============================================
# Function to Generate Windows from machine DF
# =============================================
def generate_windows_from_dfs(machine_dataframes, geometry_before_dfs, geometry_after_dfs, output_directory=None, save_csv=False, method='middle_small'):
    stats_before_springback = []
    stats_after_springback = []

    for index, (machine_df, geo_before_df, geo_after_df) in enumerate(zip(machine_dataframes, geometry_before_dfs, geometry_after_dfs)):
        stats_before = process_geometry(machine_df, geo_before_df, method)
        stats_after = process_geometry(machine_df, geo_after_df, method)

        # Convert lists of dictionaries to DataFrames
        df_before = pd.DataFrame(stats_before) if stats_before else pd.DataFrame()
        df_after = pd.DataFrame(stats_after) if stats_after else pd.DataFrame()

        # Append to the main list and check if DataFrame is empty before saving
        if not df_before.empty:
            stats_before_springback.append(df_before)
            if save_csv and output_directory:
                df_before.to_csv(os.path.join(output_directory, f'sim_{index+1}_before_springback_stats.csv'), index=False)

        if not df_after.empty:
            stats_after_springback.append(df_after)
            if save_csv and output_directory:
                df_after.to_csv(os.path.join(output_directory, f'sim_{index+1}_after_springback_stats.csv'), index=False)

    return (stats_before_springback, stats_after_springback)

def process_geometry(machine_df, geometry_df, method):
    stats = []
    window_size = {'middle_small': 0.5, 'middle_large': 2, 'middle_extra_large': 3, 'before': 1, 'after': 1}[method]

    for angle in geometry_df['Angle[degree]']:
        lower_bound, upper_bound = calculate_bounds(angle, window_size, method, machine_df)

        mask = (machine_df['BendDieBendingAngle'] >= lower_bound) & (machine_df['BendDieBendingAngle'] <= upper_bound)
        window_df = machine_df[mask]

        if not window_df.empty:
            summary_stats = {
                'Angle[degree]': angle,
                **{f'{col}_min': window_df[col].min() for col in window_df if col != 'BendDieBendingAngle'},
                **{f'{col}_max': window_df[col].max() for col in window_df if col != 'BendDieBendingAngle'},
                **{f'{col}_avg': window_df[col].mean() for col in window_df if col != 'BendDieBendingAngle'}
            }
            stats.append(summary_stats)

    return stats

def calculate_bounds(angle, window_size, method, machine_df):
    if method in ['middle_small', 'middle_large', 'middle_extra_large']:
        lower_bound = max(angle - window_size, 0)
        upper_bound = angle + window_size
    elif method == 'before':
        lower_bound = max(angle - window_size, 0)
        upper_bound = angle
        # Adjust bounds if necessary based on data availability
        if machine_df[(machine_df['BendDieBendingAngle'] >= lower_bound) & (machine_df['BendDieBendingAngle'] <= upper_bound)].empty:
            lower_bound = angle
            upper_bound = angle + 2  # Extend to 'after' if 'before' is empty
    elif method == 'after':
        lower_bound = angle
        upper_bound = angle + window_size
        # Adjust bounds if necessary based on data availability
        if machine_df[(machine_df['BendDieBendingAngle'] >= lower_bound) & (machine_df['BendDieBendingAngle'] <= upper_bound)].empty:
            lower_bound = max(angle - 2, 0)  # Extend to 'before' if 'after' is empty
            upper_bound = angle

    return lower_bound, upper_bound


# =============================================
# Function to Merge Windowed DF with Geometry DF
# =============================================
def merge_machine_with_geometry(machine_stats_before, machine_stats_after, geometry_before_dfs, geometry_after_dfs, output_directory=None, save_csv=False):
    """
    Merges windowed machine data with geometry data for each simulation and optionally saves the merged data to CSV files.

    Parameters:
    - machine_stats_before (list of DataFrames): DataFrames containing windowed stats for machine data before springback.
    - machine_stats_after (list of DataFrames): DataFrames containing windowed stats for machine data after springback.
    - geometry_before_dfs (list of DataFrames): Geometry data frames before springback.
    - geometry_after_dfs (list of DataFrames): Geometry data frames after springback.
    - output_directory (str): Directory to save the CSV files if save_csv is True.
    - save_csv (bool): Flag to determine whether to save the merged DataFrames to CSV.

    Returns:
    - Tuple of lists: (merged_before_springback, merged_after_springback)
    """
    merged_before_springback = []
    merged_after_springback = []

    # Process each simulation for before springback
    for i, (machine_df, geo_df) in enumerate(zip(machine_stats_before, geometry_before_dfs)):
        merged_df = pd.merge(machine_df, geo_df, on='Angle[degree]', how='inner')
        merged_before_springback.append(merged_df)
        if save_csv and output_directory:
            merged_df.to_csv(os.path.join(output_directory, f'merged_before_springback_{i+1}.csv'), index=False)

    # Process each simulation for after springback
    for i, (machine_df, geo_df) in enumerate(zip(machine_stats_after, geometry_after_dfs)):
        merged_df = pd.merge(machine_df, geo_df, on='Angle[degree]', how='inner')
        merged_after_springback.append(merged_df)
        if save_csv and output_directory:
            merged_df.to_csv(os.path.join(output_directory, f'merged_after_springback_{i+1}.csv'), index=False)

    return (merged_before_springback, merged_after_springback)


# =============================================
# Function to Merge all DFs into two huge ones
# =============================================

def concatenate_data_frames(merged_before, merged_after, output_directory=None, save_csv=False):
    """
    Concatenates lists of data frames into two large data frames and optionally saves them to CSV.

    Parameters:
    - merged_before (list of DataFrames): List of data frames containing merged machine and geometry data before springback.
    - merged_after (list of DataFrames): List of data frames containing merged machine and geometry data after springback.
    - output_directory (str): Directory to save the CSV files if save_csv is True.
    - save_csv (bool): Flag to determine whether to save the concatenated DataFrames to CSV.

    Returns:
    - Tuple of DataFrames: (concatenated_before, concatenated_after)
    """
    # Concatenate all DataFrames in the list for before springback
    concatenated_before = pd.concat(merged_before, ignore_index=True)

    # Concatenate all DataFrames in the list for after springback
    concatenated_after = pd.concat(merged_after, ignore_index=True)

    # Check if saving to CSV is required
    if save_csv and output_directory:
        before_path = os.path.join(output_directory, 'final_merged_before.csv')
        after_path = os.path.join(output_directory, 'final_merged_after.csv')
        concatenated_before.to_csv(before_path, index=False)
        concatenated_after.to_csv(after_path, index=False)

    return concatenated_before, concatenated_after
