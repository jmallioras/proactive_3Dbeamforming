%% Next Position Calculation Function
% This function calculates a new position given an initial location, a target location, 
% velocity, time, and a maximum angular deviation for randomization. The function uses 
% geographical coordinates and trigonometric calculations to determine the new position.

function newLocation = nextPosition(loc1, loc2, velocity, time, maxAngularDeviation)
    
    % Extract latitude and longitude coordinates from input locations
    lat1 = loc1(1);
    lon1 = loc1(2);
    lat2 = loc2(1);
    lon2 = loc2(2);

    % Earth's radius in meters
    R = 6371000; 

    % Convert latitude and longitude to radians
    lat1 = lat1 * pi / 180;
    lon1 = lon1 * pi / 180;
    lat2 = lat2 * pi / 180;
    lon2 = lon2 * pi / 180;

    % Calculate the bearing (angle) from the initial point to the target point
    y = sin(lon2 - lon1) * cos(lat2);
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1);
    bearing = atan2(y, x);

    % Add a random deviation to the bearing
    stdDev = maxAngularDeviation / 3; % Standard deviation for random deviation
    randomDeviation = randn * stdDev; % Random deviation using normal distribution
    bearing = bearing + randomDeviation; % Apply deviation to the bearing

    % Calculate the distance to be traveled (meters)
    distance = velocity * time;

    % Calculate the new latitude and longitude in radians using the haversine formula
    newLat = asin(sin(lat1) * cos(distance / R) + cos(lat1) * sin(distance / R) * cos(bearing));
    newLon = lon1 + atan2(sin(bearing) * sin(distance / R) * cos(lat1), cos(distance / R) - sin(lat1) * sin(newLat));

    % Convert the new latitude and longitude back to degrees
    newLat = newLat * 180 / pi;
    newLon = newLon * 180 / pi;

    newLocation = [newLat, newLon]; % Return the new location
end
