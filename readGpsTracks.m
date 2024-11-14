function tracks = readGpsTracks(filename)
    % Open the file
    fid = fopen(filename, 'r');
    
    % Initialize variables
    tracks = {};
    currentTrack = [];
    
    % Read the file line by line
    while ~feof(fid)
        line = fgetl(fid);
        
        % Check for empty line (track separator)
        if isempty(line) || all(isspace(line))
            if ~isempty(currentTrack)
                tracks{end+1} = currentTrack;
                currentTrack = []; % Reset for next track
            end
            continue;
        end
        
        % Parse the line (assuming tab-separated values)
        data = strsplit(line, '\t');
        
        % Skip the header or non-coordinate lines
        if length(data) < 3 || ~strcmp(data{1}, 'T')
            continue;
        end
        
        % Extract latitude and longitude
        lat = str2double(data{2});
        lon = str2double(data{3});
        
        % Append to current track
        currentTrack = [currentTrack; lat, lon];
    end
    
    % Add the last track if not empty
    if ~isempty(currentTrack)
        tracks{end+1} = currentTrack;
    end
    
    % Close the file
    fclose(fid);
end

