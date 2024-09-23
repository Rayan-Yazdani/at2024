import seaborn as sns
import matplotlib.pyplot as plt  # Importing matplotlib
import numpy as np


# Compute Correlation Matrix:
def compute_matrix(df):
    correlation_matrix = df.corr()
    return correlation_matrix

# Visualize with a Heatmap
def visualise_matrix(correlation_matrix, display_labels=False):
    
    sns.set(style='white')

    # Set the context to "talk" for larger heatmaps, adjust to "paper" or "poster" for even larger sizes
    sns.set_context("talk", font_scale=0.5)
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(18, 12))  # Adjust the figure size as necessary

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlation_matrix, mask=mask, annot=display_labels, cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # Optional: Rotate labels on the x-axis if you still want to display them
    if display_labels:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    print("Showing Plots for Each Feature Correlation:")
    plt.show()  # This will display the plot


def feature_correlation_matrix(df, display_labels):
    correlation_matrix = compute_matrix(df)
    visualise_matrix(correlation_matrix, display_labels)


# Remove Highly Correlated Features:
def remove_correlated_features(df, correlation_threshold):
    """
    Remove highly correlated and constant features from a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    correlation_threshold (float): Threshold for correlation. Features with a correlation 
                                   equal to or higher than this value will be removed.

    Returns:
    pd.DataFrame: A DataFrame with the highly correlated and constant features removed.
    """
    # Calculate the correlation matrix
    corr_matrix = compute_matrix(df)

    # Identify features that are highly correlated
    high_corr = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) >= correlation_threshold:  # Check the absolute value of correlation
                colname = corr_matrix.columns[i]
                high_corr.add(colname)

    # Remove highly correlated features
    df = df.drop(columns=high_corr)

    # Remove constant features (features with zero standard deviation)
    constant_columns = df.columns[df.nunique() <= 1]
    df = df.drop(columns=constant_columns)

    return df

# Example usage:
# filtered_data = remove_correlated_features(consolidated_data, 0.9)
