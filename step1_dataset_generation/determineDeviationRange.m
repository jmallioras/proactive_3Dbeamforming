%% Determine Deviation Range Function
% This function calculates the allowable deviation range in radians based on 
% the activity type. Different activities (Stroll, Walk, Jog, Run) have different 
% allowable deviation angles, which are converted from degrees to radians.

function devRange = determineDeviationRange(activityType)
    switch activityType
        case 1 % Stroll activity
            devRange = deg2rad(40);
        case 2 % Walk activity
            devRange = deg2rad(30);
        case 3 % Jog activity
            devRange = deg2rad(20);
        case 4 % Run activity
            devRange = deg2rad(10);
        otherwise
            error('Invalid activity type'); % Error handling for invalid input
    end
end
