import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
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

def analyze_feature_importance(df_before, df_after, target_option, n_folds=5):
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

    

    return avg_mse

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Define the mapping of simple keys to actual column names
"""
TARGET_OPTIONS = {
    1: 'before_springback_Secondary axis[mm]',
    2: 'before_springback_Main axis[mm]',
    3: 'before_springback_Out-of-roundness[-]',
    4: 'before_springback_Collapse[mm]',
    5: 'after_springback_Secondary axis[mm]',
    6: 'after_springback_Main axis[mm]',
    7: 'after_springback_Out-of-roundness[-]',
    8: 'after_springback_Collapse[mm]'
}
"""
# List of specific input features to analyze
specific_features = [
    'Diameter tube [mm]_min',
    'Wallthickness tube [mm]_min',
    'Mandrel extraction before bending end []_min',
    'Collet boost [-]_min',
    'Clearance pressure die [mm]_min'
]

def analyze_specific_features(df):
    feature_importances = pd.DataFrame(columns=specific_features)

    for target_column in TARGET_OPTIONS.values():
        # Prepare data: drop other target options and keep only the specific features and current target
        X = df.drop(list(TARGET_OPTIONS.values()), axis=1)
        X = X[specific_features]
        y = df[target_column]

        # Train the Random Forest model
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X, y)

        # Extract feature importance for the specific features
        importances = rf.feature_importances_
        feature_importances.loc[target_column] = importances

    # Convert feature importances to percentages and round
    feature_importances = (feature_importances * 100).round(2)

    # Print the feature importances table
    print("Feature Importances Table for Specific Features:")
    print(feature_importances)

    # Plotting the feature importances
    feature_importances.plot(kind='bar', figsize=(15, 10))
    plt.title('Feature Importances for Each Target Option')
    plt.ylabel('Importance (%)')
    plt.xlabel('Target Options')
    plt.xticks(rotation=45)
    plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return feature_importances

# Example usage:
# df = pd.read_csv('path_to_your_data.csv')  # Load your DataFrame
# specific_feature_importances = analyze_specific_features(df)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the mapping of simple keys to actual column names
"""
TARGET_OPTIONS = {
    1: 'before_springback_Secondary axis[mm]',
    2: 'before_springback_Main axis[mm]',
    3: 'before_springback_Out-of-roundness[-]',
    4: 'before_springback_Collapse[mm]',
    5: 'after_springback_Secondary axis[mm]',
    6: 'after_springback_Main axis[mm]',
    7: 'after_springback_Out-of-roundness[-]',
    8: 'after_springback_Collapse[mm]'
}
"""
def analyze_all_features(df):
    feature_importances_summary = pd.DataFrame()

    for key, target_column in TARGET_OPTIONS.items():
        features_df = df.drop(list(TARGET_OPTIONS.values()), axis=1)
        target_df = df[target_column]

        rf = RandomForestRegressor(random_state=42)
        rf.fit(features_df, target_df)

        importances = rf.feature_importances_
        features = features_df.columns

        feature_importances_summary[target_column] = pd.Series(importances, index=features)

    feature_importances_summary = feature_importances_summary.T
    feature_importances_summary = (feature_importances_summary * 100).round(2)

    return feature_importances_summary

def plot_top_n_features(feature_importances_summary, top_n=5):
    for target in feature_importances_summary.index:
        top_features = feature_importances_summary.loc[target].sort_values(ascending=False).head(top_n)
        plt.figure(figsize=(8, 6))
        top_features.plot(kind='bar')
        plt.title(f'Top {top_n} Features for {target}')
        plt.ylabel('Importance (%)')
        plt.xlabel('Features')
        plt.show()

def plot_average_feature_importance(feature_importances_summary):
    avg_importance = feature_importances_summary.mean(axis=0).sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    avg_importance.plot(kind='bar')
    plt.title('Average Feature Importance Across All Targets')
    plt.ylabel('Average Importance (%)')
    plt.xlabel('Features')
    plt.show()

def plot_importance_heatmap(feature_importances_summary):
    plt.figure(figsize=(14, 10))
    sns.heatmap(feature_importances_summary, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Importance Heatmap Across All Targets')
    plt.ylabel('Target Options')
    plt.xlabel('Features')
    plt.xticks(rotation=45)
    plt.show()

def plot_cumulative_importance(feature_importances_summary, top_n=5):
    for target in feature_importances_summary.index:
        sorted_importances = feature_importances_summary.loc[target].sort_values(ascending=False)
        cumulative_importance = np.cumsum(sorted_importances)
        plt.figure(figsize=(8, 6))
        cumulative_importance.head(top_n).plot(drawstyle='steps-post')
        plt.title(f'Cumulative Importance for {target}')
        plt.ylabel('Cumulative Importance (%)')
        plt.xlabel('Number of Top Features')
        plt.xticks(range(top_n), labels=sorted_importances.index[:top_n], rotation=45)
        plt.show()

def visualize_feature_importance(df):
    feature_importances = analyze_all_features(df)
    print("Select the visualization option:")
    print("1: Top-N Feature Importances")
    print("2: Average Feature Importance")
    print("3: Heatmap of Feature Importances")
    print("4: Cumulative Importance of Top-N Features")
    choice = input("Enter your choice (1-4): ")

    if choice == '1':
        top_n = int(input("Enter the number of top features to display: "))
        plot_top_n_features(feature_importances, top_n=top_n)
    elif choice == '2':
        plot_average_feature_importance(feature_importances)
    elif choice == '3':
        plot_importance_heatmap(feature_importances)
    elif choice == '4':
        top_n = int(input("Enter the number of top features for cumulative importance: "))
        plot_cumulative_importance(feature_importances, top_n=top_n)
    else:
        print("Invalid choice. Please enter a number between 1 and 4.")

# Example usage
# df = pd.read_csv('path_to_your_data.csv')  # Ensure you load your DataFrame
# visualize_feature_importance(df)
