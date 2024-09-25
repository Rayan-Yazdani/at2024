import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
import graphviz
import subprocess 

# Simplified Target Options without prefixes
TARGET_OPTIONS = {
    1: 'Secondary axis[mm]',
    2: 'Main axis[mm]',
    3: 'Out-of-roundness[-]',
    4: 'Collapse[mm]',
    5: 'Secondary axis[mm]',
    6: 'Main axis[mm]',
    7: 'Out-of-roundness[-]',
    8: 'Collapse[mm]'
}

def analyze_data(df_before, df_after, target_option, n_folds=5, output_directory=None, plot_tree=False, save_csv=False):
    if target_option not in TARGET_OPTIONS:
        raise ValueError("Invalid target option. Please choose a number from 1 to 8.")
    
    df = df_before if 1 <= target_option <= 4 else df_after
    target_column = TARGET_OPTIONS[target_option]
    # Start
    # Use a dictionary to filter out duplicates, keeping the first occurrence
    unique_targets = {}
    for key, value in TARGET_OPTIONS.items():
        if value not in unique_targets:
            unique_targets[value] = key  # Store first occurrence only
    # Now create a list of columns to drop, which are the unique target names
    columns_to_drop = list(unique_targets.keys())
    X = df.drop(columns=columns_to_drop, errors='ignore')  # Drop all unique target columns 
    # End
    # X = df.drop(columns=target_column)
    y = df[target_column]

    rf = RandomForestRegressor(random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=n_folds, scoring='neg_mean_squared_error')
    avg_mse = np.mean(-cv_scores)
    rf.fit(X, y)

    # Save X and y as CSV files
    if save_csv and output_directory:
        x_output_path = os.path.join(output_directory, "X_data.csv")
        y_output_path = os.path.join(output_directory, "y_data.csv")
        X.to_csv(x_output_path, index=False)
        y.to_csv(y_output_path, index=False)
    
    if plot_tree and output_directory:
        tree = rf.estimators_[0]
        dot_data = export_graphviz(tree, out_file=None, feature_names=X.columns, filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        
        output_path = os.path.join(output_directory, "random_forest_tree")
        graph.format = 'png'
        graph.render(output_path)
        
        print(f"Tree diagram saved as PNG at: {output_path}")

    return avg_mse
  







"""


def combine_simulations_to_csv(df, indices_series1, indices_series2, target_option):
    try:
        # Ensure the target option is valid
        if target_option not in TARGET_OPTIONS:
            raise ValueError(f"Invalid target option. Choose from {list(TARGET_OPTIONS.keys())}")
        
        target_column = TARGET_OPTIONS[target_option]

        # Ensure the indices are within the valid range
        if any(index < 0 or index >= len(df) for index in indices_series1):
            raise IndexError("One or more indices in the first series are out of range.")
        if any(index < 0 or index >= len(df) for index in indices_series2):
            raise IndexError("One or more indices in the second series are out of range.")
        
        # Combine the selected data frames for the first series
        combined_df1 = pd.concat([df[index] for index in indices_series1], ignore_index=True)
        
        # Create df_train by dropping target columns
        df_train = combined_df1.drop(columns=TARGET_OPTIONS.values(), errors='ignore')
        
        # Create df_train_target with only the target option column
        df_train_target = combined_df1[[target_column]]
        
        # Combine the selected data frames for the second series
        combined_df2 = pd.concat([df[index] for index in indices_series2], ignore_index=True)
        
        # Create df_test by dropping target columns
        df_test = combined_df2.drop(columns=TARGET_OPTIONS.values(), errors='ignore')
        
        # Create df_test_target with only the target option column
        df_test_target = combined_df2[[target_column]]
        
        return df_train, df_train_target, df_test, df_test_target
    
    except IndexError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

"""





def combine_and_train_random_forest(df, indices_series1, indices_series2, target_option):
    """
    Combine data frames at specified indices in df, train a Random Forest model, and test the model.
    
    Parameters:
    df (list of pd.DataFrame): List of data frames.
    indices_series1 (list of int): Indices of the data frames to be combined for the first series.
    indices_series2 (list of int): Indices of the data frames to be combined for the second series.
    target_option (int): The column number to be retained in the output data frames.
    
    Returns:
    tuple: A tuple containing four data frames (df_train, df_train_target, df_test, df_test_target) and the model performance metrics.
    """
    try:
        # Ensure the target option is valid
        if target_option not in TARGET_OPTIONS:
            raise ValueError(f"Invalid target option. Choose from {list(TARGET_OPTIONS.keys())}")
        
        target_column = TARGET_OPTIONS[target_option]

        # Ensure the indices are within the valid range
        if any(index < 0 or index >= len(df) for index in indices_series1):
            raise IndexError("One or more indices in the first series are out of range.")
        if any(index < 0 or index >= len(df) for index in indices_series2):
            raise IndexError("One or more indices in the second series are out of range.")
        
        # Combine the selected data frames for the first series
        combined_df1 = pd.concat([df[index] for index in indices_series1], ignore_index=True)
        
        # Create df_train by dropping target columns
        df_train = combined_df1.drop(columns=TARGET_OPTIONS.values(), errors='ignore')
        
        # Create df_train_target with only the target option column
        df_train_target = combined_df1[[target_column]]
        
        # Combine the selected data frames for the second series
        combined_df2 = pd.concat([df[index] for index in indices_series2], ignore_index=True)
        
        # Create df_test by dropping target columns
        df_test = combined_df2.drop(columns=TARGET_OPTIONS.values(), errors='ignore')
        
        # Create df_test_target with only the target option column
        df_test_target = combined_df2[[target_column]]
        
        # Train the Random Forest model
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(df_train, df_train_target.values.ravel())
        
        # Test the model
        predictions = rf_model.predict(df_test)
        
        # Calculate performance metrics
        mse = mean_squared_error(df_test_target, predictions)
        r2 = r2_score(df_test_target, predictions)
        
        print(f"Model Performance:\nMean Squared Error: {mse}\nR^2 Score: {r2}")
        
        return df_train, df_train_target, df_test, df_test_target, mse, r2
    
    except IndexError as e:
        print(e)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")

