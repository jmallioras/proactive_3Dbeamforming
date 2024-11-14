%% Ray Tracing heuristic Function
% This function selects and configures a ray tracing propagation model based on the
% specified class. The class determines the maximum number of reflections and 
% diffractions allowed for ray tracing. If an invalid class is provided, the function
% sets a flag to indicate that no valid ray tracing model exists.

function [rtpm, noRay] = properProp(class)
    % Initialize default values for reflections, diffractions, and ray existence flag
    max_ref = 0; % Maximum number of reflections
    max_dif = 0; % Maximum number of diffractions
    noRay = false; % Flag to indicate if no valid ray model is available

    % Configure propagation parameters based on the class
    switch class
        case 1
            max_ref = 0; % No reflections
            max_dif = 0; % No diffractions
        case 2
            max_ref = 1; % Allow 1 reflection
            max_dif = 0; % No diffractions
        case 3
            max_ref = 2; % Allow 2 reflections
            max_dif = 0; % No diffractions
        case 4
            max_ref = 2; % Allow 2 reflections
            max_dif = 1; % Allow 1 diffraction
        case 5
            max_ref = 3; % Allow 3 reflections
            max_dif = 1; % Allow 1 diffraction
        otherwise
            noRay = true; % Invalid class, no valid ray model
    end

    % Define the ray tracing propagation model with specified parameters
    rtpm = propagationModel("raytracing", ...
            "Method", "sbr", ...
            "MaxNumReflections", max_ref, ...
            "MaxNumDiffractions", max_dif, ...
            "AngularSeparation", "medium", ...
            "BuildingsMaterial", "concrete", ...
            "TerrainMaterial", "concrete");
end
