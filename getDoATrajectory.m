%% Get DoA Trajectory Function
% This function calculates the Direction of Arrival (DoA) trajectory for a given track.
% It sets up the transmitter (Tx) and receiver (Rx) configurations, defines antenna patterns,
% and uses ray tracing to determine the angles of departure (AoDs) at each step along the track.
% The resulting AoDs are adjusted based on antenna tilts and stored in the output array.
% These AoDs are the equivalent DoAs if the roles of transmitter and
% receiver where reversed.

function AODS = getDoATrajectory(track)
    %% Generate Antenna Mesh Grid and Gain Pattern
    % Define azimuth and elevation angles for the mesh grid
    azvec = -180:180; % Azimuth angles in degrees
    elvec = -90:90;   % Elevation angles in degrees
    [az, el] = meshgrid(azvec, elvec);
    
    % Antenna parameters for gain pattern calculation
    tilt = 0; % Elevation tilt angle
    SLA = 30; % Maximum side-lobe attenuation level (dB)
    az3dB = 65; % 3 dB beamwidth for azimuth (degrees)
    el3dB = 65; % 3 dB beamwidth for elevation (degrees)
    
    % Calculate magnitude patterns for azimuth and elevation
    azMagPattern = -min(12 * (az / az3dB).^2, SLA);
    elMagPattern = -min(12 * ((el - tilt) / el3dB).^2, SLA);
    combinedMagPattern = -min(-(azMagPattern + elMagPattern), SLA); % Combined antenna gain pattern (dB)

    %% Transmitter (Tx) Configuration
    tx = txsite("Latitude", 37.944973, "Longitude", 23.644138, "AntennaHeight", 5, ...
                "TransmitterPower", 5, "TransmitterFrequency", 27e9);
    
    % Define the antenna properties for the transmitter
    lambda = physconst("lightspeed") / tx.TransmitterFrequency; % Wavelength (m)
    antennaElement = phased.CustomAntennaElement("MagnitudePattern", combinedMagPattern);
    tx.Antenna = phased.URA("Size", [8 8], "Element", antennaElement, ...
                            "ElementSpacing", [lambda/2 lambda/2]);
    
    % Set initial antenna tilt angles
    AZIMUTH_TILT = -20; % Azimuth tilt angle (degrees)
    ELEVATION_TILT = -30; % Elevation tilt angle (degrees)
    tx.AntennaAngle = [AZIMUTH_TILT, ELEVATION_TILT];

    %% Process Track to Generate AoDs
    steps = size(track, 1); % Number of steps in the track
    AODS = []; % Initialize output array for AoDs

    for step = 1:steps
        % Extract current latitude and longitude from the track
        lat = track(step, 1);
        lon = track(step, 2);
        rx = rxsite("Latitude", lat, "Longitude", lon, "AntennaHeight", 2); % Define receiver site
        
        % Start with initial propagation class
        class = 2;
        [rtpm, noRay] = properProp(class); % Get initial propagation parameters
        ray = raytrace(tx, rx, rtpm); % Perform ray tracing
        
        % Adjust propagation class if no line of sight (LoS) is found
        % While no ray is found, move to the next class of the heuristic to
        % improve chances of identifying rays.
        while isempty(ray{1, 1})
            class = class + 1;
            [rtpm, noRay] = properProp(class);
            if noRay == true
                break; % Exit loop if no valid ray is found
            end
            ray = raytrace(tx, rx, rtpm); % Retry ray tracing with updated parameters
        end
        
        if noRay == true
            continue; % Skip to next step if no valid ray was found
        end 
        
        % Extract Angle of Departure (AoD) from the ray
        aod = ray{1}.AngleOfDeparture;
        steeringaz = wrapTo180(aod(1) - tx.AntennaAngle(1)); % Adjust azimuth angle
        steeringel = aod(2) - tx.AntennaAngle(2); % Adjust elevation angle
        aod = [steeringaz; steeringel];
        aod = nnConv(aod); % Apply any necessary conversions (assumed external function)
        
        % Store AoDs in output array
        AODS = [AODS; aod(1), aod(2)];
    end
end
