%% Get Next Track Point Function
% This function determines the next index in a predefined path based on the 
% current location. It checks if the current position is within a specified 
% radius around the previous and next track points to decide the next step. 
% A boolean flag is used to indicate if the end of the track is reached.

function [nextIdx, endOfTrack] = getNextTrackPoint(currentNextIdx, currentLoc, predefinedPath)
    endOfTrack = false; % Initialize end of track flag
    
    % Calculate the distance (traceRadius) between the previous and next predefined track points
    traceRadius = sqrt((predefinedPath(currentNextIdx, 1) - predefinedPath(currentNextIdx - 1, 1))^2 + ...
                       (predefinedPath(currentNextIdx, 2) - predefinedPath(currentNextIdx - 1, 2))^2);
    
    % Calculate the distance (distanceFromPrevious) between the current position and the previous track point
    distanceFromPrevious = sqrt((currentLoc(1) - predefinedPath(currentNextIdx - 1, 1))^2 + ...
                                (currentLoc(2) - predefinedPath(currentNextIdx - 1, 2))^2);

    % Determine the next index based on distance comparison
    if traceRadius > distanceFromPrevious
        nextIdx = currentNextIdx; % Stay at the current index
    else
        nextIdx = currentNextIdx + 1; % Move to the next index
    end
    
    % Check if the proposed next index exceeds the track bounds
    if nextIdx > size(predefinedPath, 1)
       endOfTrack = true; % Set end of track flag to true
    end
end
