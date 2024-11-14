%% Track Production Script
% This script processes predefined GPS traces (e.g., from gpsvisualizer.com)
% to generate realistic movement tracks for pedestrians or vehicles. 
% The produceTrack() function creates a random, realistic movement pattern 
% based on these traces. The resulting tracks are stored and saved as a 
% dataset.

% Note: Ensure the OSM map file and predefined trace file exist in the
% specified path.

% Set OSM map file of the area of interest
map_file = "peiraias.osm"; 

% Set file containing the predefined GPS traces
predefined_traces_file = 'peiraias_pedestrian_tracks.txt'; 

% Specify track type: pedestrian (false) or vehicle (true)
vehicle_track = false; 

% Define the output dataset name
dataset_filename = 'dummy_test'; 

% Set the number of tracks to be produced
n_paths = 10;

% Open the specified OSM map file
viewer = siteviewer("Basemap", "openstreetmap", "Buildings", map_file); 

% Load the predefined GPS traces
tracks = readGpsTracks(predefined_traces_file);

% Initialize a cell array to store tracks of variable length
paths = cell(n_paths, 1); % Preallocate memory for generated paths

% Loop to generate and store tracks
for path_id = 1:n_paths
    % Generate a random movement track based on predefined traces
    paths{path_id} = produceTrack(tracks, viewer, vehicle_track);

    % Display progress and periodically save data
    if mod(path_id, 1000) == 0
        disp(['Generated paths: ', num2str(path_id)]); 
        
        % Define path saving parameters
        folderName = 'Paths/'; 
        folderPath = fullfile(pwd, folderName); 
        
        % Create the folder if it does not exist
        if ~exist(folderPath, 'dir')
            mkdir(folderPath);
        end
        
        % Construct filename and save data
        fileName = dataset_filename + string(path_id) + '.mat'; 
        fullPath = fullfile(folderPath, fileName); % Generate complete save path
        save(fullPath, 'paths'); % Save generated paths
    end
end

% Save the final dataset
folderName = 'Paths/'; % Folder for storing paths
folderPath = fullfile(pwd, folderName); % Construct save path

% Create folder if it does not exist
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end

% Final dataset save operation
fileName = dataset_filename + string(n_paths) + '.mat'; % Filename for final dataset
fullPath = fullfile(folderPath, fileName); % Complete path for file
save(fullPath, 'paths'); % Save paths to .mat file
disp('Tracks generated!');
disp("Saved at: "+fullPath);