# =========================================================
# Data Preparation (Machine Data)
# This will save each simulation data in a single dataframe
# =========================================================
import os
from pathlib import Path
print()
print("\033[94mData Preparation (Machine Data)\033[0m")
# Import the process_simulations function from Data_Preparation module
from Data_Preparation import process_simulations

# Define your parameters
base_directory = Path(__file__).resolve().parent.parent / 'Simulation Data'
output_directory = Path(__file__).resolve().parent.parent / 'output'
output_directory.mkdir(parents=True, exist_ok=True)
sim_start = 1  # Starting simulation number
sim_end = 162  # Ending simulation number 162
print_option = 'nprint'  # Set to 'print' to export the DataFrame to CSV, or 'nprint' to not export
time_option = 'deltime'  # Set to 'time' to keep the time column, or 'deltime' to delete it

# Specify which simulation number to plot (Time column should not be eliminated)
plot_sim_num = None  # Plot the simulation number, write None if you do not want to plot

# Call the function
simulation_dataframes = process_simulations(base_directory, output_directory, sim_start, sim_end, print_option, time_option, plot_sim_num)
print(f"\033[94mtotal number of {sim_end} simulations were saved in a variable named \033[91msimulation_dataframes\033[0m")

# This part of the code will confirm the process of simulations and print out the reult
from collections import defaultdict
row_count_groups = defaultdict(list)# Dictionary to store the simulation indices grouped by their row count
for i, df in enumerate(simulation_dataframes):# Group data frames by their row count
    row_count = len(df)
    row_count_groups[row_count].append(sim_start + i)
green = "\033[92m"# ANSI escape code for green color
reset = "\033[0m"
for row_count, indices in row_count_groups.items():# Print the summary
    simulations_str = ", ".join(f"{green}{index}{reset}" for index in indices)
    print(f"Simulations {simulations_str} are processed and have {row_count} rows.")

# Termination of Data Preparation Part
print("\033[1;38;5;111mFinished Simulation Data Preparation\033[0m")
print()

# ====================================================================================
# Data Preparation (Geometry Data)
# This will put the geometry data (before and after springback) in separate dataframes
# ====================================================================================
print("\033[94mData Preparation (Geometry Data)\033[0m")
from Data_Preparation import load_and_plot_geometry

# Define your parameters
plot = None # Set to True if you want to plot set to None if you don't
plot_sims=[1, 60, 100] # Simulation geometries you would like to print

# Call the function to load data and optionally plot
geometry_before_dataframes, geometry_after_dataframes = load_and_plot_geometry(
    base_directory, sim_start, sim_end, plot, plot_sims)

# Optional: Print a summary or handle the dataframes as needed
print(f"\033[94mLoaded {len(geometry_before_dataframes)} simulations for geometry before springback in a dataframe named \033[91mgeometry_before_dataframes\033[0m")
print(f"\033[94mLoaded {len(geometry_after_dataframes)} simulations for geometry after springback in a dataframe named \033[91mgeometry_after_dataframes\033[0m")

print("\033[1;38;5;111mFinished Geometry Data Preparation\033[0m")
print()
# print(simulation_dataframes)  # Add this in your generate_windows_from_dfs function to check actual column names




# =================
# Window Generation
# =================
print("\033[94mWindow Generation\033[0m")
from Data_Preparation import generate_windows_from_dfs
method = 'simple'
save_csv = False # set to True if you want to have the CSV file for the windowed data otherwise False

# Explain and choose windowing method
print("Select a windowing method by entering the corresponding number:")
print("""
1 - Middle Small: ±0.5° around each angle, adjusts at data limits.
2 - Middle Large: ±2° around each angle for wider context.
3 - Middle Extra Large: ±3° for maximum context.
4 - Before: 1° before each angle, or 2° after if limited data before.
5 - After: 1° after each angle, or 2° before if limited data after.
""")
method_input = input("Enter your choice (1-5): ")
method_dict = {
    '1': 'middle_small',
    '2': 'middle_large',
    '3': 'middle_extra_large',
    '4': 'before',
    '5': 'after'
}
method = method_dict.get(method_input, None)


# Validate and generate windows
if method:
    stats_before_springback, stats_after_springback = generate_windows_from_dfs(
        simulation_dataframes, geometry_before_dataframes, geometry_after_dataframes,
        output_directory=output_directory, save_csv=save_csv, method=method)
    print("Finished Processing and Generating Statistical Windows.")
else:
    print("Invalid method selected. Please run the script again with a valid number (1-5).")

print(f"\033[94mLoaded {len(stats_before_springback)} simulations for combination of simulation data and geometry before springback in a dataframe named \033[91mstats_before_springback\033[0m")
print(f"\033[94mLoaded {len(stats_after_springback)} simulations for combination of simulation data and geometry after springback in a dataframe named \033[91mstats_after_springback\033[0m")
print("\033[1;38;5;111mFinished Window Creation\033[0m")
print()


# ======================================
# Merge Windowed Data with Geometry Data
# ======================================
print("\033[94mMerge Windowed Data with Geometry Data\033[0m")
from Data_Preparation import merge_machine_with_geometry, concatenate_data_frames

save_csv=False # set to True if you want a CSV output

merged_before, merged_after = merge_machine_with_geometry(
    stats_before_springback, stats_after_springback, geometry_before_dataframes, geometry_after_dataframes,
    output_directory=output_directory, save_csv=save_csv
)
print("\033[31mFinished merging machine data with geometry data.\033[0m")

# Concatenate all individual DataFrames into two large DataFrames and optionally save them
save_csv=False # Set to True if you want CSV files of the whole Dataframes, otherwise False

concatenated_before, concatenated_after = concatenate_data_frames(
    merged_before, merged_after,
    output_directory=output_directory, save_csv=save_csv
)
print(f"\033[94mMerged all {len(stats_before_springback)} data in \033[91mstats_before_springback\033[0m to a dataframe named \033[91mconcatenated_before\033[0m ")
print(f"\033[94mMerged all {len(stats_after_springback)} data in \033[91mstats_after_springback\033[0m to a dataframe named \033[91mconcatenated_after\033[0m ")
print("\033[1;38;5;111mFinished Merging All Data Frames\033[0m")
print()


# ===================
# Feature Correlation
# ===================
print("\033[94mFeature Correlation\033[0m")
from Feature_Correlation import feature_correlation_matrix, remove_correlated_features
feature_correlation_matrix(concatenated_before, display_labels=True)  # Compute correlation matrix and visualise it with Heatmap
concatenated_before = remove_correlated_features(concatenated_before, 0.98)  # removes high correlated features which has the ration of 1
print("\033[31mRemoved high correlated features\033[0m")
feature_correlation_matrix(concatenated_before, display_labels=True)  # Compute correlation matrix and visualise it with Heatmap again
print("\033[1;38;5;111mFinished Feature Correlation for before Springback\033[0m")
print()
feature_correlation_matrix(concatenated_after, display_labels=True)  # Compute correlation matrix and visualise it with Heatmap
concatenated_after = remove_correlated_features(concatenated_after, 0.98)  # removes high correlated features which has the ration of 1
print("\033[31mRemoved high correlated features\033[0m")
feature_correlation_matrix(concatenated_after, display_labels=True)  # Compute correlation matrix and visualise it with Heatmap again
print("\033[1;38;5;111mFinished Feature Correlation for after Springback\033[0m")
print()


# ==============
# Model Training
# ==============
print("\033[94mModel Training\033[0m")
from model_analysis import analyze_data, TARGET_OPTIONS

# Print options for the user
print("Select a target option:")
for key, value in TARGET_OPTIONS.items():
    # Differentiate display based on before or after springback
    description = "before springback" if key <= 4 else "after springback"
    print(f"{key}: {description} {value}")

# User selects the target option and number of folds
target_option = int(input("Enter your choice (1-8): "))
n_folds = int(input("Enter the number of folds for cross-validation: "))
plot_request = input("Do you want to plot the Random Forest tree? Enter 'yes' or 'no': ").lower() == 'yes'
save_csv = input("Do you want to save training data and target data as csv? Enter 'yes' or 'no': ").lower() == 'yes'

# Call the function and handle potential errors
try:
    avg_mse = analyze_data(concatenated_before, concatenated_after, target_option, n_folds, output_directory, plot_request, save_csv)
    print(f"The average Mean Squared Error across {n_folds} folds is: {avg_mse}")
    if plot_request:
        print(f"Random Forest tree saved to: {output_directory}")
except ValueError as e:
    print(e)

print("\033[1;38;5;111mFinished Model Training\033[0m")
print()

"""
# ===========================================================
# Code for training on some simulations and testing on others
# ===========================================================
print("\033[94mTraining on some simulations and testing on others\033[0m")
from model_analysis import combine_and_train_random_forest, TARGET_OPTIONS
# Print options for the user to select the target column
print("Select a target option:")
for key, value in TARGET_OPTIONS.items():
    # Differentiate display based on before or after springback
    description = "before springback" if key <= 4 else "after springback"
    print(f"{key}: {description} {value}")
target_option = int(input("Enter your choice (1-8) for the target column: "))
# Get the simulation numbers for training and testing
train_sim_input = input("Enter the simulation numbers for training (separated by commas): ")
test_sim_input = input("Enter the simulation numbers for testing (separated by commas): ")
train_sim_nums = [int(num.strip()) for num in train_sim_input.split(',')]
test_sim_nums = [int(num.strip()) for num in test_sim_input.split(',')]
test_together_input = input("Do you want to test the simulations together? (yes/no): ").lower() == 'yes'
df = merged_before if target_option <=4 else merged_after
# Call the function
combine_and_train_random_forest(df, train_sim_nums, test_sim_nums, target_option)
# last checked
###
# Call the function
mse_results = train_test_multiple_simulations(df, train_sim_nums, test_sim_nums, test_together_input, target_option)
for i, mse in enumerate(mse_results, 1):
    print(f"The Mean Squared Error for test simulation {test_sim_nums[i-1] if not test_together_input else 'all'} is: {mse}")

"""

# ===========================================================
# Infinite loop for training on some simulations and testing on others
# ===========================================================
while True:
    print("\033[94mTraining on some simulations and testing on others\033[0m")
    from model_analysis import combine_and_train_random_forest, TARGET_OPTIONS

    # Get the simulation numbers for training and testing
    train_sim_input = input("Enter the simulation numbers for training (separated by commas): ")
    test_sim_input = input("Enter the simulation numbers for testing (separated by commas): ")
    train_sim_nums = [int(num.strip()) for num in train_sim_input.split(',')]
    test_sim_nums = [int(num.strip()) for num in test_sim_input.split(',')]

    # Loop through target options 5 to 8
    for target_option in range(5, 9):
        # Use merged_after since target options 5-8 correspond to "after springback"
        df = merged_after

        # Call the function for each target option
        print(f"\nRunning for Target Option {target_option}:")
        combine_and_train_random_forest(df, train_sim_nums, test_sim_nums, target_option)

    # Optionally, add a break condition if you want to exit the loop
    continue_loop = input("Do you want to run another set of simulations? (yes/no): ").lower() == 'yes'
    if not continue_loop:
        print("Exiting the loop.")
        break
"""
while True:
    print("\033[94mTraining on some simulations and testing on others\033[0m")
    from model_analysis import combine_and_train_random_forest, TARGET_OPTIONS

    # Print options for the user to select the target column
    print("Select a target option:")
    for key, value in TARGET_OPTIONS.items():
        # Differentiate display based on before or after springback
        description = "before springback" if key <= 4 else "after springback"
        print(f"{key}: {description} {value}")
    
    target_option = int(input("Enter your choice (1-8) for the target column: "))
    
    # Get the simulation numbers for training and testing
    train_sim_input = input("Enter the simulation numbers for training (separated by commas): ")
    test_sim_input = input("Enter the simulation numbers for testing (separated by commas): ")
    train_sim_nums = [int(num.strip()) for num in train_sim_input.split(',')]
    test_sim_nums = [int(num.strip()) for num in test_sim_input.split(',')]
    
    test_together_input = False
    
    df = merged_before if target_option <= 4 else merged_after
    
    # Call the function
    combine_and_train_random_forest(df, train_sim_nums, test_sim_nums, target_option)
 

    # Optionally, add a break condition if you want to exit the loop
    continue_loop = input("Do you want to run another simulation? (yes/no): ").lower() == 'yes'
    if not continue_loop:
        print("Exiting the loop.")
        break
"""


# ===========================
# Feature importance analysis
# ===========================
print("\033[94mFeature Importance Analysis\033[0m")
from Feature_Importance import analyze_feature_importance, TARGET_OPTIONS, analyze_specific_features, visualize_feature_importance


# Print options for the user
#print("Select a target option:")
#for key, value in TARGET_OPTIONS.items():
#    # Differentiate display based on before or after springback
#    description = "before springback" if key <= 4 else "after springback"
#    print(f"{key}: {description} {value}")

# User selects the target option and number of folds
#target_option = int(input("Enter your choice (1-8): "))
#n_folds = int(input("Enter the number of folds for cross-validation: "))

# Call the function and handle potential errors
#try:
#    avg_mse = analyze_feature_importance(concatenated_before, concatenated_after, target_option, n_folds)
#    print(f"The average Mean Squared Error across {n_folds} folds is: {avg_mse}")
#    if plot_request:
#        print(f"Random Forest tree saved to: {output_directory}")
#except ValueError as e:
#    print(e)
# specific_feature_importances = analyze_specific_features(df)
print("Features for Before Springback:")
visualize_feature_importance(concatenated_before)
print("Features for After Springback:")
visualize_feature_importance(concatenated_after)
print("Setup Features for Before Springback:")
analyze_specific_features(concatenated_before)
print("Setup Features for After Springback:")
analyze_specific_features(concatenated_after)



