%% Create DoA Trajectories Dataset Script
% This script imports predefined movement tracks and generates Direction of 
% Arrival (DoA) trajectories for each track using a specified map. The resulting 
% DoA tracks are stored and saved periodically as part of a dataset. The script 
% checks for consistency between the tracks and the generated DoA data. The
% getAods() function is used to retrieve the DoA trajectories for each track.

% Note: Ensure the OSM map file and track data file exist in the specified path.

% Import the map for visualization
viewer = siteviewer("Basemap", "openstreetmap", "Buildings", "peiraias.osm");

% Set final dataset name
final_dataset_name =  'dummy_doa_tracks';

% Load the predefined tracks from the specified file
trackdata = load("Paths/dummy_test10.mat");
tracks = trackdata.paths;
n_tracks = size(tracks, 1); % Get the number of tracks
track_DOAS = cell(n_tracks, 1); % Preallocate cell array for DoA data

% Starting index for generating tracks
start_idx = 0;

% Loop through each track to generate DoA trajectories
for track_id = start_idx + 1:n_tracks
    track = tracks{track_id}; % Extract the current track
    track_DOAS{track_id} = getDoATrajectory(track); % Generate DoA trajectory for the track

    % Display progress and periodically save data
    if mod(track_id, 250) == 0
        disp(['Processed tracks: ', num2str(track_id)]); 
        
        % Define path saving parameters
        folderName = 'Dataset/';
        folderPath = fullfile(pwd, folderName);
        
        % Create the folder if it does not exist
        if ~exist(folderPath, 'dir')
            mkdir(folderPath);
        end
        
        % Construct filename and save data
        fileName = final_dataset_name + string(track_id) + '.mat'; 
        fullPath = fullfile(folderPath, fileName); % Generate complete save path
        save(fullPath, 'track_DOAS'); % Save DoA tracks
    end
end

% Check for paths with mismatched sizes (no rays found)
idxs = [];
for i = 1:n_tracks
    if size(track_DOAS{i}, 1) ~= size(tracks{i}, 1)
        idxs = [idxs; i]; % Record problematic indices
    end
end

% Calculate the size differences for problematic paths
dif_idxs = zeros(length(idxs), 1);
for j = 1:length(idxs)
    i = idxs(j);
    dif_idxs(j) = abs(size(track_DOAS{i}, 1) - size(tracks{i}, 1));
end

% Define final dataset saving parameters
folderName = 'Dataset/';
folderPath = fullfile(pwd, folderName);

% Create the folder if it does not exist
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end

% Construct the final filename and save the dataset
fileName = final_dataset_name + string(n_tracks / 1e3) + 'k.mat'; 
fullPath = fullfile(folderPath, fileName); % Generate complete save path
save(fullPath, 'track_DOAS'); % Save the final DoA dataset

% Close the parallel pool if it exists
pool = gcp('nocreate');
if ~isempty(pool)
    delete(pool); % Close the parallel pool
end
disp("New dataset created containing "+n_tracks+" DoA trajectories!");
disp("Saved at: "+fullPath);