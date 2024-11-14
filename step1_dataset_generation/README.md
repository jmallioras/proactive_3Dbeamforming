# Generating DoA Trajectory Tracks

This guide explains how to generate Direction of Arrival (DoA) trajectory tracks using the provided MATLAB scripts. The process leverages a predefined track file created with the `gpsvisualizer.com` tool and an OpenStreetMap (OSM) file for geographic context and visualization.

## Prerequisites

- **MATLAB** version higher than 2023b with the Phased Array System Toolbox, Mapping Toolbox, Antenna Toolbox.
- A predefined track file in `.txt` format, generated using `gpsvisualizer.com`.
- An OSM map file (e.g., `peiraias.osm`) for geographic visualization and context.

## Input Files and Their Purpose

### 1. OSM Map File (`peiraias.osm`)
The OSM file provides geographic data for the area of interest. It is used with the `siteviewer` function to visualize tracks and facilitate ray-tracing.

### 2. Predefined Track File (`peiraias_pedestrian_tracks.txt`)
This text file contains predefined GPS tracks in a specific format. The file is structured with latitude and longitude coordinates and is read using the `readGpsTracks` function. The expected format is:

Each entry consists of:
- `type`: Identifier 
- `latitude`: Latitude coordinate
- `longitude`: Longitude coordinate
- `name` and `desc` (optional): Additional details, if present.

You can draw your own GPS traces, using [gpsvisualizer.com](https://www.gpsvisualizer.com/draw/) and export the file once you have are finished.

## Usage Instructions
There are two main scripts that you need to execute to generate the final DoA trajectory dataset. The first one is `track_production_dataset.m` where realistic movement tracks are generated based on the predefined GPS traces you provided. The second, is `doa_trajectory_dataset.m` where for each of the movement tracks, the DoAs (azimuth, elevation) are collected at each timestep using ray-tracing with an efficient heuristic method.

### Step 1: Reading the Predefined Tracks and creating realistic movement patterns

In `track_production_dataset.m`, change the necessary paths to your local files and define the dataset name(containing the user movement tracks) as well as the total number of random tracks you whish to generate based on the predefined GPS traces in the `.txt` file:

```matlab
    % Set OSM map file of the area of interest
    map_file = "peiraias.osm"; 

    % Set file containing the predefined GPS traces
    predefined_traces_file = 'peiraias_pedestrian_tracks.txt';

    % Define the output dataset name
    dataset_filename = 'dummy_test'; 
```
Aferwards, run the script. This will generate a set of movement tracks in the `\Paths` directory. 


### Step 2: Generating DoA Trajectories
To generate DoA trajectories for a given track, use the `doa_trajectory_dataset.m` script. Make sure to define the name of the final dataset (containing the DoA trajectories).

```matlab
% Set final dataset name
final_dataset_name =  'dummy_doa_tracks';
```
