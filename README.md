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

## Introduction

Rotary draw bending (RDB) is a key technique used to shape thin-walled tubes, but accurately predicting the resulting geometry can be complex. This project explores the use of Random Forest regression models to predict geometrical outcomes, providing insights that can improve the design and optimization of the tools used in the bending process. Below, is the cross section image of a tube bending machine

![IMG_5ADD3E36BDE5-1](https://github.com/user-attachments/assets/3c2a7ddd-b378-4602-ac1d-36ac7af87a9f)


## Features

- Random Forest Regression Model: Provides geometry predictions based on machine forces and movements.
- Finite Element Simulation: Data from 162 simulated tube-bending processes used for model training.
- Feature Importance Analysis: Identifies which process parameters most influence the accuracy of the model.
- Data-Driven Insights: Helps improve bending tool design and reduce manual adjustments.

## Installation

To install and run the project, follow the steps below:

1. Clone the repository: `git clone https://github.com/Rayan-Yazdani/at2024.git`  
   Then, navigate to the project folder: `cd at2024`

2. Create a virtual environment: `python3 -m venv venv`  
   Activate the virtual environment: `source venv/bin/activate`  
   (On Windows: `venv\Scripts\activate`)

3. Install the required dependencies: `pip install -r requirements.txt`

## Data

The simulation data used for this project is stored in the `Simulation Data` folder. It includes the following:

- **Forces and Movements**: Time-series data on the forces and movements of tools during the bending process.
  <img width="501" alt="Picture1" src="https://github.com/user-attachments/assets/77b1a681-b8a6-4ea8-a3ec-9b2ffbb5ddae">

- **Geometries**: Tube geometry data both before and after springback, including secondary and main axes, out-of-roundness, and collapse.
  ![IMG_0666](https://github.com/user-attachments/assets/19c4bb4e-4e33-4eda-a72c-d16b7a011806)

- **Process Parameters**: Important parameters such as tube diameter, wall thickness, mandrel retraction timing, collet boost, and pressure die clearance.

The full dataset is available here: [Simulation Data](Simulation%20Data/).

## Usage

The primary Python script for running the model is `Main2.py`. Here’s how to use it:

1. **Data Preparation**: Ensure that the simulation data is available in the correct directory.
2. **Run the script**: Execute the following command to run the main Python file:
   
   `python3 Code/Main2.py`

3. **Window Selection**: You can select one of the windows
   - Type 1: Middle Small
   - Type 2: Middle Large
   - Type 3: Middle Extra Large
   - Type 4: Before
   - Type 5: After
  
4. **Feature Correlation**: The correlation of the features will be shown based on the selected window

5. **Model Training**: Based on the selected window a model will be trained but before that we should select the target and the number of folds for cross validation (usually 5), next, the MSE for predicting that target would be generated. This section also has the ability to visualize the random forest. a dot format file will be generated, if you had difficulty visualising it, you can use the following code to convert it into a PDF format:
   `dot -Tpdf -o random_forest_tree.pdf random_forest_tree`
7. **Training on Some Simulations And testing on others**: the next part of the code is designed if we want to train and test on some specific simulations instead of all 162 together

8. **Feature Importance Analysis**: the next part of the code is for feature importance. basically select a target data and it would generate the importance of all features affecting that predicting target.

## Results

The Random Forest model successfully predicts tube geometries with RMS errors under 0.19 mm for a tube diameter of 22 mm. The model identified the timing of mandrel retraction and collet boost as the most critical factors in achieving accurate predictions.


