%% Initialize Start Point Function
% This function selects a random starting point from a predefined path and 
% interpolates between two consecutive points to create a smooth starting location.
% The predefined path is an Nx2 array containing latitude and longitude coordinates.

function [startPoint, nextIndex] = initializeStartPoint(predefinedPath)
    % Get the number of points in the predefined path
    numPoints = size(predefinedPath, 1); % Total number of points in the path
    midIndex = floor(numPoints / 2); % Calculate the midpoint index of the path

    % Select a random index up to the mid point (excluding the last point)
    startIndex = randi([1, midIndex]); % Random starting index for interpolation

    % Extract the coordinates of the chosen point and its consecutive point
    point1 = predefinedPath(startIndex, :); % First point coordinates
    point2 = predefinedPath(startIndex + 1, :); % Consecutive point coordinates

    % Perform linear interpolation with a random factor t between 0 and 1
    t = rand(); % Random interpolation factor
    
    % Calculate the interpolated latitude and longitude
    interpolatedLat = point1(1) + t * (point2(1) - point1(1));
    interpolatedLon = point1(2) + t * (point2(2) - point1(2));

    % Set the starting point for the new path
    startPoint = [interpolatedLat, interpolatedLon]; % Interpolated starting point

    % Indicate the next track index
    nextIndex = startIndex + 1; % Next index in the track
end
