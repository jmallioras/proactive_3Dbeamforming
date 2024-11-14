%% Produce Track Function
% This function generates a realistic random movement pattern based on the 
% geographical area, obstacles, and predefined paths in the "tracks" argument.
% It uses "vehicle_flag" to differentiate between pedestrian and vehicle movements.

function new_path = produceTrack(tracks, viewer, vehicle_flag)
    viewer = viewer; % Assign viewer object
    
    % If pedestrian tracks, make them bi-directional
    tracks = appendFlipped(tracks); % Function to append flipped tracks for bidirectional movement
    
    %% Production Details
    max_recorded_steps = 50; % Maximum number of recorded steps for a path
    min_steps = 7; % Minimum steps required to form a valid track
    max_steps = 100; % Maximum allowable steps in a single track
    
    % Details for calculating new positions
    max_on_roof_attempts = 10; % Maximum attempts to correct paths on buildings
    
    if vehicle_flag == true
        haltChance = 0.15; % Probability of halting or slowing down for vehicles
    else
        haltChance = 0.05; % Probability of halting or slowing down for pedestrians
    end
    
    turnDecRate = 1; % Rate of velocity decrease on turns
    maxTurn = 180; % Maximum turn angle in degrees
    time = 1; % Time in seconds for track updates
    step_sample = 1; % Sampling rate for recording steps
    
    % Select a random predefined track from the available tracks
    track = tracks{1, randi([1, size(tracks, 2)])}; % Randomly pick a track
    step = 0; % Initialize step count
    
    % Randomly select activity type for movement pattern
    activityType = selectActivity(size(track, 1)); % Choose activity type based on track size
    
    if vehicle_flag == true
        maxDeviation = deg2rad(10); % Smaller deviation for vehicles
    else
        maxDeviation = determineDeviationRange(activityType); % Determine deviation based on activity type
    end

    INIT_MAX_DEVIATION = maxDeviation; % Save initial deviation value
    
    if vehicle_flag == true
        velocityRange = [2.7, 13.8]; % Typical velocity range for vehicles (m/s)
    else
        velocityRange = determineVelocityRange(activityType); % Determine pedestrian velocity range
    end

    meanVelocity = rand * (velocityRange(2) - velocityRange(1)) + velocityRange(1); % Random initial mean velocity
    
    recorded_traces = 0; % Initialize trace recording counter
    
    % Main loop to produce a new track
    while recorded_traces < min_steps  
        on_roof_counter = 0; % Counter for "on-roof" correction attempts
        
        % Randomly select initial point in the track
        [currentLoc, nextTraceIdx] = initializeStartPoint(track); % Initialize starting point
        endOfTrack = false; % Flag for end of track
        
        % Update current coordinates for position tracking
        rx = rxsite("Latitude", currentLoc(1), "Longitude", currentLoc(2), "AntennaHeight", 1);
        avg_elevation = elevation(rx); % Determine average elevation for comparison
        
        new_path = []; % Initialize new path array
        recorded_traces = 0; % Reset trace count
        
        % Loop to generate steps in the track
        for step = 2:max_steps
            if on_roof_counter >= max_on_roof_attempts
                break; % Stop if max on-roof attempts reached
            end
            
            % Record coordinates at specified intervals
            if mod(step, step_sample) == 0
                recorded_traces = recorded_traces + 1;
                new_path = [new_path; currentLoc(1), currentLoc(2)]; % Append coordinates to path
                if recorded_traces == max_recorded_steps
                    break; % Stop if max recorded steps reached
                end
            end
            
            % Get next point in the track
            [nextTraceIdx, endOfTrack] = getNextTrackPoint(nextTraceIdx, currentLoc, track);
            if endOfTrack == true
                break; % End track if last point reached
            end
            
            nextPredefinedPoint = track(nextTraceIdx, :); % Fetch next predefined point
            
            % Calculate turning angle
            if nextTraceIdx + 1 <= size(track, 1)
                theta = min(turnIndication(currentLoc, nextPredefinedPoint, track(nextTraceIdx + 1, :)), maxTurn);
            else
                theta = 0; % No turn if at end of track
            end
            
            turningFactor = (((maxTurn - abs(theta)) / maxTurn))^turnDecRate; % Adjust velocity for turn
            vel = @(r) 0.1 * (r <= haltChance) + meanVelocity * (r > haltChance); % Calculate velocity with halt chance
            velocity = vel(rand) * turningFactor; % Apply turning factor to velocity
            
            % Correction loop for "on-building" detection
            on_a_building = true;
            while on_a_building == true
                newLoc = nextPosition(currentLoc, nextPredefinedPoint, velocity, time, maxDeviation); % Calculate new position
                rx = rxsite("Latitude", newLoc(1), "Longitude", newLoc(2), "AntennaHeight", 2); % Update position site
                
                % Check elevation to detect if on building
                if elevation(rx) >= avg_elevation + 2
                    on_roof_counter = on_roof_counter + 1;
                    maxDeviation = maxDeviation - 10; % Reduce deviation for correction
                    if on_roof_counter >= max_on_roof_attempts
                        break; % Stop if max attempts exceeded
                    end
                else
                    avg_elevation = ((step - 1) / step) * (avg_elevation + (1 + (step / max_steps)) * elevation(rx) / (step - 1));
                    on_a_building = false; % Successfully corrected
                    maxDeviation = INIT_MAX_DEVIATION; % Reset max deviation
                end
            end
            currentLoc = newLoc; % Update current location for next iteration
        end
    end
end
