%% Append Flipped Cells Function
% This script defines two functions: 
% 1. appendFlipped - This function takes an input cell array and returns a new cell array 
% that contains both the original elements and their flipped versions.
% 2. flipArray - This function flips an input array vertically.

function outputCells = appendFlipped(inputCells)
    % Count total number of cells in the input array
    numCells = numel(inputCells);
    
    % Preallocate output array to hold original and flipped cells
    outputCells = cell(1, 2 * numCells);
    
    % Copy original cells and append flipped versions
    for i = 1:numCells
        outputCells{i} = inputCells{i}; % Copy original cell to output
        outputCells{i + numCells} = flipArray(inputCells{i}); % Store flipped version of the original cell
    end
end

function flippedArray = flipArray(inputArray)
    % Flip the array vertically
    flippedArray = flipud(inputArray);
end

