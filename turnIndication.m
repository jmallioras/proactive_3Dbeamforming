%% Turn Indication Function
% This function calculates the angle between three positions (current, next, 
% and next-next positions) using the law of cosines. It returns the turning 
% angle in degrees, which can indicate the change of direction at a given position.

function value = turnIndication(currentPos, nextPos, next_nextPos)
    % Calculate the distances between the three positions
    a = sqrt((currentPos(1) - nextPos(1))^2 + (currentPos(2) - nextPos(2))^2); % Distance between current and next position
    b = sqrt((currentPos(1) - next_nextPos(1))^2 + (currentPos(2) - next_nextPos(2))^2); % Distance between current and next-next position
    c = sqrt((nextPos(1) - next_nextPos(1))^2 + (nextPos(2) - next_nextPos(2))^2); % Distance between next and next-next position
    
    % Calculate the turning angle using the law of cosines
    value = acosd(((a^2) + (b^2) - (c^2)) / (2 * a * b));
end
