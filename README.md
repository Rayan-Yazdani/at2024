# Tube Geometry Prediction Using Random Forest

## Overview

This repository provides a machine learning-based approach to predict the geometry of steel tubes during rotary draw bending using Random Forest regression. The model leverages finite element simulation data to optimize tool configurations and improve bending accuracy.

## Abstract

**This project implements Random Forest regression to predict the geometry of bent tubes in the rotary draw bending process. The model is trained on data from 162 simulations, which include machine forces, movements, and tube geometries. The analysis identifies critical parameters that impact prediction accuracy, such as mandrel retraction timing and collet boost.**

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Data](#data)
5. [Usage](#usage)
6. [Results](#results)
7. [Graphs and Visualizations](#graphs-and-visualizations)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction

Rotary draw bending (RDB) is a key technique used to shape thin-walled tubes, but accurately predicting the resulting geometry can be complex. This project explores the use of Random Forest regression models to predict geometrical outcomes, providing insights that can improve the design and optimization of the tools used in the bending process.

## Features

- Random Forest Regression Model: Provides geometry predictions based on machine forces and movements.
- Finite Element Simulation: Data from 162 simulated tube-bending processes used for model training.
- Feature Importance Analysis: Identifies which process parameters most influence the accuracy of the model.
- Data-Driven Insights: Helps improve bending tool design and reduce manual adjustments.

## Installation

To install and run the project, follow the steps below:

1. Clone the repository: `git clone https://github.com/Rayan-Yazdani/at2024.git`  
   Then, navigate to the project folder: `cd at2024`

2. Create a virtual environment: `python -m venv venv`  
   Activate the virtual environment: `source venv/bin/activate`  
   (On Windows: `venv\Scripts\activate`)

3. Install the required dependencies: `pip install -r requirements.txt`

## Data

The simulation data used for this project is stored in the `Simulation Data` folder. It includes the following:

- **Forces and Movements**: Time-series data on the forces and movements of tools during the bending process.
- **Geometries**: Tube geometry data both before and after springback, including secondary and main axes, out-of-roundness, and collapse.
- **Process Parameters**: Important parameters such as tube diameter, wall thickness, mandrel retraction timing, collet boost, and pressure die clearance.

The full dataset is available here: [Simulation Data](Simulation%20Data/).

## Usage

The Python code to execute the Random Forest regression model can be found in the `code` folder. You can run the model with the following steps:

1. Import the module: `from code.predict import predict_geometry`

2. Load data and predict geometry: `predicted_geometry = predict_geometry("Simulation Data/input_data.csv")`

Make sure the `Simulation Data` folder is in the correct directory for the code to access the input data.

## Results

The Random Forest model successfully predicts tube geometries with RMS errors under 0.19 mm for a tube diameter of 22 mm. The model identified the timing of mandrel retraction and collet boost as the most critical factors in achieving accurate predictions.

## Graphs and Visualizations

To better illustrate the results, you can include visualizations generated from the model's predictions:

- **Prediction vs Actual Geometry**: A comparison between the predicted and actual geometries before and after springback.
- **Feature Importance Chart**: A bar chart showing which process parameters had the highest impact on prediction accuracy.

You can create these graphs using matplotlib or another plotting library, and save them in an `images` folder.

Example code to generate and save the graph:

import matplotlib.pyplot as plt

# Example: Generate and save a feature importance plot
plt.bar(feature_names, feature_importances)
plt.title("Feature Importance")
plt.savefig("images/feature_importance.png")

Here’s where you can include these images in the README:

![Prediction vs Actual Geometry](images/prediction_vs_actual.png)  
*Figure 1: Comparison of predicted vs actual geometry.*

![Feature Importance](images/feature_importance.png)  
*Figure 2: Process parameters ranked by importance.*

(*Remember to generate and save these graphs in the `images` folder.*)

## Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
