# Integrating the transformer model in a Proactive Beamforming Simulation in Urban Environment

The MATLAB script `proactive_beamforming.m` simulates a wireless communication system in an urban environment, focusing on proactive beamforming for enhanced signal strength and interference suppression.
It utilizes the small transformer model we trained in STEP 2 to predict the direction of arrival (DoA) of signals, then evaluates the signal strength, 
and compares the performance of the model. The script contains lines for integration of the RNN models and the big TNN model presented in the paper:  
*"A Novel Neural Network Approach to Proactive 3-D Beamforming" [IEEE Xplore](https://ieeexplore.ieee.org/document/10750053).*

## Important Notes
- To succesfully execute `proactive_beamforming.m`, make sure to place the script accompanied by the extra functions `proactive_models.m`, `RNN.m` and the directory `model_inference` in the same working directory with the files used in STEP 1. This is because the script utilizes various of the helper functions included there.
- The directory `model_inference` contains the necessary python scripts and pre-trained models to utilize these models for inference through the MATLAB function`proactive_models.m`.
- You can uncomment the lines related to the models you have available for comparison.

## Script description
- `RNN.m`: Invokes a python script that uses a pre-trained LSTM beamformer presented in *"Enhancing Adaptive Beamforming in 3-D Space Through Self-Improving Neural Network Techniques" [IEEE Xplore](https://ieeexplore.ieee.org/document/10438855).* . This model takes as input three DoAs considering the first one to be the DoA of the desired signal and the rest of undesired interferences and returns the beamforming weight vector to perform zero-forcing beamforming.
- `proactive_models.m` : Used to perform inference on the pre-trained DoA forecasting models by executing the relative python scripts.

## Features

- **Urban Environment Simulation**: Utilizes real map data to simulate an urban setting.
- **Antenna Pattern Generation**: Creates custom antenna patterns for the transmitter.
- **Beam Steering and Null-Steering**: Implements beam steering to focus on the desired user and null-steering to suppress interference.
- **Machine Learning Integration**: Incorporates pre-trained models (e.g., TNN-small) for predicting future DoAs.
- **Signal Strength and SIR Measurement**: Evaluates performance by measuring signal strength and Signal-to-Interference Ratio (SIR).
- **Data Visualization**: Provides graphical representations of tracks, antenna patterns, signal strengths, and prediction errors.

## Requirements

- **MATLAB** R2020a or later with:
  - Antenna Toolbox
  - Phased Array System Toolbox
  - Communications Toolbox
- **Data Files**:
  - `test_pedestrian_paths.mat` (place in the `Paths` directory)
  - `peiraias.osm` (map data file)
- **Python Environment** (if using external Python scripts for machine learning inference)

## Usage

1. **Setup**:
   - Ensure all required MATLAB toolboxes are installed.
   - Place `test_pedestrian_paths.mat` in the `Paths` directory.
   - Ensure `peiraias.osm` or any other .osm file is accessible by the script.
   - Configure the Python environment in MATLAB using `pyenv` if necessary.

2. **Run the Script**:
   - Open the script in MATLAB.
   - Execute the script to start the simulation.

3. **Simulation Flow**:
   - Initializes the environment and antenna configurations.
   - Randomly selects tracks for the desired user and interferences.
   - Performs beamforming adjustments based on real and predicted DoAs.
   - Calculates and compares signal strengths and SIRs.
   - Generates plots and visualizations of the results.

## Functions Overview

- **Antenna and Environment Setup**: Defines the antenna patterns and transmitter site.
- **Track Assignment**: Loads and assigns tracks to the desired user and interferences.
- **Beamforming Techniques**:
  - **Beam Steering**: Focuses the antenna beam towards the desired user.
  - **Null-Steering with RNN**: Uses RNN-ZF (the pre-trained LSTM that performs zero-forcing beamforming) to create nulls towards interferences.
- **Proactive Beamforming**: Uses machine learning models to predict future DoAs and adjust beamforming proactively.
- **Data Visualization**: Plots signal strengths, SIRs, and prediction errors.

## Notes

- Ensure all required data files and models are properly set up before running the simulation.
- The performance of machine learning models may vary based on configurations.
- For any issues or questions, feel free to contact the project maintainers.

---

**Disclaimer**: This simulation is intended for educational and research purposes. The performance may vary based on the data and environment configurations.
