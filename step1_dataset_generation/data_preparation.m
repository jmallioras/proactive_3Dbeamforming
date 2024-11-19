% Read Track from Dataset Folder
trackdata= load("Dataset/2april_peiraias_doa_tracks20000.mat");
pathsMatrix = trackdata.track_DOAS;

% Assuming time step
dt = 1;

% Assume your matrix is named 'pathsMatrix'
numPaths = size(pathsMatrix, 1); % Number of paths in the dataset

% Initialize an empty cell array to hold the processed data
processedData = cell(numPaths, 1);

for i = 1:numPaths
    % Extract the current path (Nx2 matrix)
    currentPath = pathsMatrix{i,1};
    if size(currentPath,1)<10
        continue;
    end
    
    % Calculate the differences in elevation and azimuth to get velocities
    azimuthVelocity = zeros(size(currentPath, 1) - 1, 1);
    elevationVelocity = diff(currentPath(:,2)) / dt; 
    for j = 1:length(azimuthVelocity)
        % Calculate azimuth difference accounting for circular nature
        diffAzimuth = mod(currentPath(j+1,1) - currentPath(j,1) + 180, 360) - 180;
        azimuthVelocity(j) = diffAzimuth / dt;
    end
    
    % Calculate acceleration by finding the difference in velocities
    elevationAcceleration = diff(elevationVelocity) / dt; % Pad with zero for the last element
    azimuthAcceleration = diff(azimuthVelocity) / dt; % Pad with zero for the last element
    
    % Preallocate the new (N-1)x6 list for the current path
    newData = zeros(length(azimuthAcceleration), 6);
    
    % Populate the new dataset
    newData(:,1:2) = currentPath(3:end, :); % DOAs (excluding the first two)
    newData(:,3:4) = [azimuthVelocity(2:end, :), elevationVelocity(2:end, :)]; % Velocities  (excluding the first one)
    newData(:,5:6) = [azimuthAcceleration, elevationAcceleration]; % Accelerations
    
    % Store the processed data
    processedData{i} = newData;
end

 % Identify non-empty cells
nonEmptyCells = ~cellfun(@isempty, processedData);

% Keep only non-empty cells
processedData = processedData(nonEmptyCells);


% processedData now contains the processed dataset
% Get the current working directory
folderName = 'Dataset/';
folderPath = fullfile(pwd, folderName);

% Check if the folder exists, if not, create it
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end


% Define the file name
fileName = '2april_processed_tracks20k.mat'; 
% Full path for the file
fullPath = fullfile(folderPath, fileName);


% Save the array to the file
save(fullPath, 'processedData');


