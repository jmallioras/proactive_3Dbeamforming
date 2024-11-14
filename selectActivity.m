%% Select Activity Function
% This function selects an activity type based on the size of the input track.
% Depending on the track size, it assigns a corresponding activity index, which
% represents different movement types: Walk, Jog, or Run.

function activityIdx = selectActivity(track_size)
    if track_size < 10
        % If the track size is less than 10, assign "Walk" activity
        activityIdx = 2;
    elseif track_size < 20
        % If the track size is between 10 and 19, randomly select "Walk" or "Jog"
        activityIdx = randi([2, 3]);
    elseif track_size >= 20
        % If the track size is 20 or more, randomly select "Walk", "Jog", or "Run"
        activityIdx = randi([2, 4]);
    end
end
