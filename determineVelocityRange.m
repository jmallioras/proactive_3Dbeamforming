%% Determine Velocity Range Function
% This function returns a velocity range (in m/s) based on the given activity type.
% Different activities (Stroll, Walk, Jog, Run) have distinct velocity ranges.

function velocityRange = determineVelocityRange(activityType)
    switch activityType
        case 1 % Stroll activity
            velocityRange = [0.6, 0.8]; % Velocity range for strolling
        case 2 % Walk activity
            velocityRange = [0.8, 1.5]; % Velocity range for walking
        case 3 % Jog activity
            velocityRange = [1.5, 4]; % Velocity range for jogging
        case 4 % Run activity
            velocityRange = [4, 10]; % Velocity range for running
        otherwise
            error('Invalid activity type'); % Handle invalid activity types
    end
end
