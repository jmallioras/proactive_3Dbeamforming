%This MATLAB script simulates a wireless communication system in an urban environment,
%focusing on proactive beamforming for enhanced signal strength and interference suppression. It uses machine learning
%models to predict the direction of arrival (DoA) of signals,
%evaluates signal strength, and compares various models' performance. Key functionalities include data visualization,
%antenna pattern generation, and signal strength measurements under different scenarios.%}

% Import map
viewer = siteviewer("Basemap","openstreetmap", "Buildings","peiraias.osm");

%% Create a mesh grid of the operational range of the antenna
azvec = -180:180; % Azimuth angles (deg)
elvec = -90:90; % Elevation angles (deg)
tilt = 0;
[az,el] = meshgrid(azvec,elvec);
SLA = 30; % Maximum side-lobe level attenuation (dB)
az3dB = 65; % 3 dB beamwidth in azimuth (deg)
el3dB = 65; % 3 dB beamwidth in elevation (deg)
azMagPattern = -min(12*(az/az3dB).^2,SLA);
elMagPattern = -min(12*((el-tilt)/el3dB).^2,SLA);
combinedMagPattern = -min(-(azMagPattern + elMagPattern),SLA); % Relative antenna gain (dB)

% Define Tx and Rx(s) location
tx = txsite("Latitude",37.944973,"Longitude", 23.644138,"AntennaHeight",5,...
    "TransmitterPower",5, ...
    "TransmitterFrequency",27e9);

% Define Tx and Rx antennas
lambda = physconst("lightspeed")/tx.TransmitterFrequency; % Wavelength (m)
antennaElement = phased.CustomAntennaElement("MagnitudePattern",combinedMagPattern);
tx.Antenna = phased.URA("Size",[8 8], ...
    "Element",antennaElement, ...
    "ElementSpacing",[lambda/2 lambda/2]);

AZIMUTH_TILT = -20;
ELEVATION_TILT = -30;
tx.AntennaAngle = [AZIMUTH_TILT, ELEVATION_TILT];
steeringVector = phased.SteeringVector("SensorArray",tx.Antenna);

% Read Track from Path Folder
trackdata= load("Paths/test_pedestrian_paths5k.mat");
all_tracks = trackdata.paths;

%%
% Assign tracks for desired user and interferences

random_idx_SOI = randi(size(all_tracks,1));
disp("(DESIRED) Selected random track (id): "+string(random_idx_SOI));
tmp = all_tracks(random_idx_SOI);
track_SOI = tmp{1,1};
%%
random_idx_SOA1 = randi(size(all_tracks,1));
disp("(UNDESIRED) Selected random track (id): "+string(random_idx_SOA1));
tmp = all_tracks(random_idx_SOA1);
track_SOA1 = tmp{1,1};
%%
random_idx_SOA2 = randi(size(all_tracks,1));
disp("(UNDESIRED) Selected random track (id): "+string(random_idx_SOA2));
tmp = all_tracks(random_idx_SOA2);
track_SOA2 = tmp{1,1};


%%
START_STEP=1;
figure;

% Plot the Desired User path
geoplot(track_SOI(START_STEP:end,1), track_SOI(START_STEP:end,2), 'DisplayName', 'Desired User', ...
    'LineStyle', '-', 'Marker', '+', 'Color', 'b', 'LineWidth', 2, 'MarkerSize', 3);
hold on;


% Plot the Interference #1 path
geoplot(track_SOA1(START_STEP:end,1), track_SOA1(START_STEP:end,2), 'DisplayName', 'Interference #1', ...
    'LineStyle', '-', 'Marker', '+', 'Color', 'black', 'LineWidth', 2, 'MarkerSize', 3);
% Mark the start and end points without adding them to the legend
geoscatter(track_SOA1(START_STEP,1), track_SOA1(START_STEP,2), 50, 'g', 'filled', 'HandleVisibility', 'off');
geoscatter(track_SOA1(end,1), track_SOA1(end,2), 50, 'y', 'filled', 'HandleVisibility', 'off');

% Plot the Interference #2 path
geoplot(track_SOA2(START_STEP:end,1), track_SOA2(START_STEP:end,2), 'DisplayName', 'Interference #2', ...
    'LineStyle', '-', 'Marker', '+', 'Color', 'red', 'LineWidth', 2, 'MarkerSize', 3);
% Mark the start and end points without adding them to the legend
geoscatter(track_SOA2(START_STEP,1), track_SOA2(START_STEP,2), 50, 'g', 'filled', 'HandleVisibility', 'off');
geoscatter(track_SOA2(end,1), track_SOA2(end,2), 50, 'y', 'filled', 'HandleVisibility', 'off');
% Mark the start and end points
geoscatter(track_SOI(START_STEP,1), track_SOI(START_STEP,2), 50, 'g', 'filled', 'DisplayName', 'Start of Track');
geoscatter(track_SOI(end,1), track_SOI(end,2), 50, 'y', 'filled', 'DisplayName', 'End of Track');
% Mark the Base station with a larger and bolder marker
geoscatter(37.944973, 23.644138, 100, 'm', 'd', 'filled', 'LineWidth', 4, 'DisplayName', 'Base station');

% Set up the basemap
basemapName = "openstreetmap";
x = 37.94386;
y = 23.64672;
z = 18;
url = "b.tile.openstreetmap.org/${z}/${x}/${y}.png";
copyright = char(uint8(169));
attribution = copyright + "OpenStreetMap contributors";
addCustomBasemap(basemapName, url, "Attribution", attribution);

geobasemap(basemapName); % Sets the basemap to 'streets'
title('New track created based on predefined path');
legend
hold off;


%%
% Observations needed
predict_every = 5+2;

% Select track
steps = min([size(track_SOI,1),size(track_SOA1,1),size(track_SOA2,1)]);

% LISTS TO SAVE SIGNAL STRENGTH
MODEL_NAME = {'TNN-small', 'TNN-big', 'LSTM-MTO', 'LSTM-MTM', 'GRU-MTO', 'GRU-MTM'};
N_MODELS = 6; % Updated the number of models
SS_SV = zeros(steps,1);
SS_RNN = zeros([steps 7]); % Adjusted for 6 models
SS_CHEB = zeros([steps 7]); % Adjusted for 6 models

SIR_SV = zeros(steps,1);
SIR_RNN = zeros([steps 7]); % Adjusted for 6 models
SIR_CHEB = zeros([steps 7]); % Adjusted for 6 models

SOI_DOAS_MTLB = [];
SOA1_DOAS_MTLB = [];
SOA2_DOAS_MTLB = [];

SOI_DOAS = [];
SOA1_DOAS = [];
SOA2_DOAS = [];

% Updated lists to distinguish between models
SOI_PRED_TNNS = [];
SOI_PRED_TNNB = [];
SOI_PRED_LSTM_MTO =  [];
SOI_PRED_LSTM_MTM =  [];
SOI_PRED_GRU_MTO = [];
SOI_PRED_GRU_MTM = [];

SOA1_PRED_TNNS = [];
SOA1_PRED_TNNB = [];
SOA1_PRED_LSTM_MTO =  [];
SOA1_PRED_LSTM_MTM =  [];
SOA1_PRED_GRU_MTO = [];
SOA1_PRED_GRU_MTM = [];

SOA2_PRED_TNNS = [];
SOA2_PRED_TNNB = [];
SOA2_PRED_LSTM_MTO =  [];
SOA2_PRED_LSTM_MTM =  [];
SOA2_PRED_GRU_MTO = [];
SOA2_PRED_GRU_MTM = [];

NLOS_SOI = zeros(steps,1);
NLOS_SOA1 = zeros(steps,1);
NLOS_SOA2 = zeros(steps,1);
proactive_flag = false;

clearMap(viewer)
show(tx)
for step=1:steps

    disp("Step "+step+":")

    % Desired signal
    rx = rxsite("Latitude",track_SOI(step,1),"Longitude",track_SOI(step,2), ...
    "AntennaHeight",2);

    % Include two interferences
    rx1 = rxsite("Latitude",track_SOA1(step,1),"Longitude",track_SOA1(step,2), ...
        "AntennaHeight",2);
    rx2 = rxsite("Latitude",track_SOA2(step,1),"Longitude",track_SOA2(step,2), ...
        "AntennaHeight",2);


    if rx.elevation>26
        disp("On roof");
    end

    % Try with the first class
    class = 1;
    [rtpm, noRay] = properProp(class);

    % Get DoAs of SoAs
    ray = raytrace(tx,rx,rtpm);
    SoA1_ray = raytrace(tx,rx1,rtpm);
    SoA2_ray = raytrace(tx,rx2,rtpm);



    % If receiver is not within line of sight, increase reflections and
    % diffractions according to properProp
    while isempty(ray{1,1}) | isempty(SoA1_ray{1,1}) |isempty(SoA2_ray{1,1})
        class = class+1;
        %disp("Class: "+string(class))
        [rtpm, noRay] = properProp(class);
        if noRay==true
            break;
        end
        ray = raytrace(tx,rx,rtpm);
        SoA1_ray = raytrace(tx,rx1,rtpm);
        SoA2_ray = raytrace(tx,rx2,rtpm);
    end
    if noRay==true
        continue;
    end

    if ray{1}(1,1).LineOfSight==0
        NLOS_SOI(step) =  1;
    end
    if SoA1_ray{1}(1,1).LineOfSight==0
        NLOS_SOA1(step) =1;
    end

    if SoA2_ray{1}(1,1).LineOfSight==0
        NLOS_SOA2(step) = 1;
    end
    %raytrace(tx,rx,rtpm)

    try
        % ============== Beam steering =============== %
        % Get AoD
        aod = ray{1}.AngleOfDeparture;
        steeringaz = wrapTo180(aod(1)-tx.AntennaAngle(1));
        steeringel = aod(2)-tx.AntennaAngle(2);
        aod = [steeringaz;steeringel];

        % % Use Beam Steering to Enhance Received Power
        sv = steeringVector(tx.TransmitterFrequency,[steeringaz;steeringel]);
        tx.Antenna.Taper = conj(sv);

        % Observe pattern and signal strength
        ss = sigstrength(rx,tx,rtpm);
        ss1 = sigstrength(rx1,tx,rtpm);
        ss2 = sigstrength(rx2,tx,rtpm);
        sir_SV = 10*log10(10^(ss/10)/(10^(ss1/10)+10^(ss2/10)));
        disp("(SV) Received power with beam steering: " + ss + " dBm")
        disp("(SV) Received power interference 1: " + ss1 + " dBm |" + ss2 +" dBm")
        disp("(SV) SIR: " + sir_SV + " dBm")

        % Interferences
        SoA1 = SoA1_ray{1}.AngleOfDeparture;
        SoA1 = [wrapTo180(SoA1(1)-tx.AntennaAngle(1)); SoA1(2)-tx.AntennaAngle(2)];
        SoA2 = SoA2_ray{1}.AngleOfDeparture;
        SoA2 = [wrapTo180(SoA2(1)-tx.AntennaAngle(1)); SoA2(2)-tx.AntennaAngle(2)];

        % Update MTLB buffers
        SOI_DOAS_MTLB = [SOI_DOAS_MTLB; aod(1) aod(2)];
        SOA1_DOAS_MTLB = [SOA1_DOAS_MTLB; SoA1(1) SoA1(2)];
        SOA2_DOAS_MTLB = [SOA2_DOAS_MTLB; SoA2(1) SoA2(2)];


        % ================== Proactive models ==================== %

        % Transform DoAs for NNs
        real_SOI = nnConv(aod);
        real_SOA1 = nnConv(SoA1);
        real_SOA2 = nnConv(SoA2);

        if step>1 && length(SOI_DOAS)>=predict_every
            % Proactive On-Off
            proactive_flag = true;

            % Prepare observations for prediction
            SOI_path  = SOI_DOAS(length(SOI_DOAS)-predict_every+1:end,:);
            SOA1_path = SOA1_DOAS(length(SOA1_DOAS)-predict_every+1:end,:);
            SOA2_path = SOA2_DOAS(length(SOA2_DOAS)-predict_every+1:end,:);

            % Get predictions
            [pred_SOI, pred_SOA1, pred_SOA2]  = proactive_models(SOI_path, SOA1_path, SOA2_path);

            % Print predictions
            %{
            % SOI
            disp(newline+"Real SOI(az, el): "+string(real_SOI(1))+","+string(real_SOI(2)))
            disp("TNN-small prediction (az, el):"+string(pred_SOI(1,1))+","+string(pred_SOI(1,2)))
            disp("TNN-big prediction (az, el):"+string(pred_SOI(2,1))+","+string(pred_SOI(2,2)))
            disp("LSTM-MTO prediction (az, el):"+string(pred_SOI(3,1))+","+string(pred_SOI(3,2)))
            disp("LSTM-MTM prediction (az, el):"+string(pred_SOI(4,1))+","+string(pred_SOI(4,2)))
            disp("GRU-MTO prediction (az, el):"+string(pred_SOI(5,1))+","+string(pred_SOI(5,2)))
            disp("GRU-MTM prediction (az, el):"+string(pred_SOI(6,1))+","+string(pred_SOI(6,2)))

            % SOA1
            disp(newline+"Real SOA1(az, el): "+string(real_SOA1(1))+","+string(real_SOA1(2)))
            disp("TNN-small prediction (az, el):"+string(pred_SOA1(1,1))+","+string(pred_SOA1(1,2)))
            disp("TNN-big prediction (az, el):"+string(pred_SOA1(2,1))+","+string(pred_SOA1(2,2)))
            disp("LSTM-MTO prediction (az, el):"+string(pred_SOA1(3,1))+","+string(pred_SOA1(3,2)))
            disp("LSTM-MTM prediction (az, el):"+string(pred_SOA1(4,1))+","+string(pred_SOA1(4,2)))
            disp("GRU-MTO prediction (az, el):"+string(pred_SOA1(5,1))+","+string(pred_SOA1(5,2)))
            disp("GRU-MTM prediction (az, el):"+string(pred_SOA1(6,1))+","+string(pred_SOA1(6,2)))

            % SOA2
            disp(newline+"Real SOA2(az, el): "+string(real_SOA2(1))+","+string(real_SOA2(2)))
            disp("TNN-small prediction (az, el):"+string(pred_SOA2(1,1))+","+string(pred_SOA2(1,2)))
            disp("TNN-big prediction (az, el):"+string(pred_SOA2(2,1))+","+string(pred_SOA2(2,2)))
            disp("LSTM-MTO prediction (az, el):"+string(pred_SOA2(3,1))+","+string(pred_SOA2(3,2)))
            disp("LSTM-MTM prediction (az, el):"+string(pred_SOA2(4,1))+","+string(pred_SOA2(4,2)))
            disp("GRU-MTO prediction (az, el):"+string(pred_SOA2(5,1))+","+string(pred_SOA2(5,2)))
            disp("GRU-MTM prediction (az, el):"+string(pred_SOA2(6,1))+","+string(pred_SOA2(6,2)))
            %}
            % Update buffers
            SOI_DOAS =  [SOI_DOAS; real_SOI(1) real_SOI(2)];
            SOA1_DOAS = [SOA1_DOAS; real_SOA1(1) real_SOA1(2)];
            SOA2_DOAS = [SOA2_DOAS; real_SOA2(1) real_SOA2(2)];

            SOI_PRED_TNNS = [SOI_PRED_TNNS; pred_SOI(1,1) pred_SOI(1,2)];
            %SOI_PRED_TNNB = [SOI_PRED_TNNB; pred_SOI(2,1) pred_SOI(2,2)];
            %SOI_PRED_LSTM_MTO =  [SOI_PRED_LSTM_MTO; pred_SOI(3,1) pred_SOI(3,2)];
            %SOI_PRED_LSTM_MTM =  [SOI_PRED_LSTM_MTM; pred_SOI(4,1) pred_SOI(4,2)];
            %SOI_PRED_GRU_MTO = [SOI_PRED_GRU_MTO; pred_SOI(5,1) pred_SOI(5,2)];
            %SOI_PRED_GRU_MTM = [SOI_PRED_GRU_MTM; pred_SOI(6,1) pred_SOI(6,2)];

            SOA1_PRED_TNNS = [SOA1_PRED_TNNS; pred_SOA1(1,1) pred_SOA1(1,2)];
            %SOA1_PRED_TNNB = [SOA1_PRED_TNNB; pred_SOA1(2,1) pred_SOA1(2,2)];
            %SOA1_PRED_LSTM_MTO =  [SOA1_PRED_LSTM_MTO; pred_SOA1(3,1) pred_SOA1(3,2)];
            %SOA1_PRED_LSTM_MTM =  [SOA1_PRED_LSTM_MTM; pred_SOA1(4,1) pred_SOA1(4,2)];
            %SOA1_PRED_GRU_MTO = [SOA1_PRED_GRU_MTO; pred_SOA1(5,1) pred_SOA1(5,2)];
            %SOA1_PRED_GRU_MTM = [SOA1_PRED_GRU_MTM; pred_SOA1(6,1) pred_SOA1(6,2)];

            SOA2_PRED_TNNS = [SOA2_PRED_TNNS; pred_SOA2(1,1) pred_SOA2(1,2)];
            %SOA2_PRED_TNNB = [SOA2_PRED_TNNB; pred_SOA2(2,1) pred_SOA2(2,2)];
            %SOA2_PRED_LSTM_MTO =  [SOA2_PRED_LSTM_MTO; pred_SOA2(3,1) pred_SOA2(3,2)];
            %SOA2_PRED_LSTM_MTM =  [SOA2_PRED_LSTM_MTM; pred_SOA2(4,1) pred_SOA2(4,2)];
            %SOA2_PRED_GRU_MTO = [SOA2_PRED_GRU_MTO; pred_SOA2(5,1) pred_SOA2(5,2)];
            %SOA2_PRED_GRU_MTM = [SOA2_PRED_GRU_MTM; pred_SOA2(6,1) pred_SOA2(6,2)];
        else
            SOI_DOAS =  [SOI_DOAS; real_SOI(1) real_SOI(2)];
            SOA1_DOAS = [SOA1_DOAS; real_SOA1(1) real_SOA1(2)];
            SOA2_DOAS = [SOA2_DOAS; real_SOA2(1) real_SOA2(2)];
        end

        % Prepare RNN input
        nn_input = [real_SOI(2), real_SOA1(2), real_SOA2(2), real_SOI(1), real_SOA1(1), real_SOA2(1)];
        nn_input = round(nn_input,1);

        % Get RNN output
        nn_sv = RNN(nn_input);

        % Use Null-Steering to Enhance Received Power
        tx.Antenna.Taper = nn_sv;

        % Observe pattern and signal strength
        %pattern(tx,"Transparency",0.6)
        %raytrace(tx,rx,rtpm)
        ss_NN = sigstrength(rx,tx,rtpm);
        ss1_NN = sigstrength(rx1,tx,rtpm);
        ss2_NN = sigstrength(rx2,tx,rtpm);
        sir_RNN = 10*log10(10^(ss_NN/10)/(10^(ss1_NN/10)+10^(ss2_NN/10)));
        disp("(RNN) Received power with beam steering: " + ss_NN + " dBm")
        disp("(RNN) Received power interference 1: " + ss1_NN + " dBm |2:" + ss2_NN +" dBm")
        disp("(RNN) SIR: " + sir_RNN + " dBm")

        % PROACTIVE BF
        if proactive_flag == true
            % Using each proactive model
            for model_idx=1:N_MODELS
                % Prepare NN input
                rnn_input = [pred_SOI(model_idx,2), pred_SOA1(model_idx,2), pred_SOA2(model_idx,2), pred_SOI(model_idx,1), pred_SOA1(model_idx,1), pred_SOA2(model_idx,1)];
                rnn_input = round(rnn_input,1);

                % Get RNN output
                nn_sv = RNN(rnn_input);

                % Use Null-Steering to Enhance Received Power
                tx.Antenna.Taper = nn_sv;

                % Observe pattern and signal strength
                %pattern(tx,"Transparency",0.6)
                %raytrace(tx,rx,rtpm)
                proactive_ss_NN = sigstrength(rx,tx,rtpm);
                proactive_ss1_NN = sigstrength(rx1,tx,rtpm);
                proactive_ss2_NN = sigstrength(rx2,tx,rtpm);
                proactive_sir_RNN = 10*log10(10^(proactive_ss_NN/10)/(10^(proactive_ss1_NN/10)+10^(proactive_ss2_NN/10)));
                SS_RNN(step,1+model_idx) = proactive_ss_NN;
                SIR_RNN(step,1+model_idx) = proactive_sir_RNN;
            end
        end

        % Save stats
        % Save DoAs
        SS_SV(step) = ss;
        SS_RNN(step,1) = ss_NN;
        SIR_SV(step) = sir_SV;
        SIR_RNN(step,1) = sir_RNN;

    catch ME
        fprintf('An error occurred: %s\n', ME.message);
        disp('No ray');
        continue;
    end


end

pattern(tx,"Transparency",1)
%%
% Plotting
START_STEP = 8;
% Create a new figure
figure;
step_count = linspace(1,size(SIR_SV,1),size(SIR_SV,1));

% Plot the three lists
hold on;
plot(SIR_SV(START_STEP:end), 'k--', 'DisplayName','Beam-steering','LineWidth',3);
plot(SIR_RNN(START_STEP:end,1), 'r', 'DisplayName','(Real DoAs) + RNN-NS','LineWidth',3);
plot(SIR_RNN(START_STEP:end,2), 'b', 'DisplayName','(TNN-small DoAs) + RNN-NS','LineWidth',3);
%{
plot(SIR_RNN(START_STEP:end,3), 'b--', 'DisplayName','(TNN-big DoAs) + RNN-NS','LineWidth',3);
plot(SIR_RNN(START_STEP:end,4), 'm', 'DisplayName','(LSTM-MTO DoAs) + RNN-NS','LineWidth',3);
plot(SIR_RNN(START_STEP:end,5), 'm--', 'DisplayName','(LSTM-MTM DoAs) + RNN-NS','LineWidth',3);
plot(SIR_RNN(START_STEP:end,6), 'g', 'DisplayName','(GRU-MTO DoAs) + RNN-NS','LineWidth',3);
plot(SIR_RNN(START_STEP:end,7), 'g--', 'DisplayName','(GRU-MTM DoAs) + RNN-NS','LineWidth',3);
%}
% Adding labels and title
xlabel('Step Count', 'FontSize', 16);  % Increased font size for x-axis label
ylabel('SIR (dB)', 'FontSize', 16);    % Increased font size for y-axis label
title('SIR Performance Comparison', 'FontSize', 18);  % Increased font size for title

% Set font size of the axes
set(gca, 'FontSize', 14);  % Set axes ticks font size

% Set x-axis limits
xlim([1,steps-START_STEP+1]);

% Add a legend (keep default font size for the legend)
legend;

% Hold off to stop adding more plots to this figure
hold off;



%%
% Plotting
START_STEP = 1;
steps = size(SIR_SV, 1);  % Assuming 'steps' is defined somewhere before

% Create a new figure
figure;
step_count = linspace(1, steps, steps);

% Plot the three lists
hold on;
xlim([1, steps - START_STEP + 1]);
ylim([5, 53]);
% Plot SIR_SV and SIR_RNN(:, 1) from START_STEP to the end
plot(SIR_SV(START_STEP:end), 'LineStyle', '-','Marker','o','Color', '#F1D400', 'DisplayName', '(Real DoA) + Beam-steering', 'LineWidth', 2);
plot(SIR_RNN(START_STEP:end, 1),'LineStyle', '-','Marker','+', 'Color','k' ,'DisplayName', '(Real DoAs) + RNN-ZF', 'LineWidth', 2);

% Plot SIR_RNN(:, 2:7) starting from the first non-zero value with correct colors and vertical lines
colors = ['g', 'b', 'c', 'm', 'y', [0.9290 0.6940 0.1250]];
legend_entries = {'Beam-steering', '(Real DoAs) + RNN-ZF', '(TNN-small DoAs) + RNN-ZF', ...
                  '(TNN-big DoAs) + RNN-ZF', '(LSTM-MTO DoAs) + RNN-ZF', '(LSTM-MTM DoAs) + RNN-ZF', ...
                  '(GRU-MTO DoAs) + RNN-ZF', '(GRU-MTM DoAs) + RNN-ZF'};


plot(predict_every+1:steps, SIR_RNN(8:end, 2),'Color','r', 'LineWidth', 2, 'DisplayName', legend_entries{3});
%{
plot(predict_every+1:steps, SIR_RNN(8:end, 3),'LineStyle','-.', 'Color','r',  'LineWidth', 2, 'DisplayName', legend_entries{4});
plot(predict_every+1:steps, SIR_RNN(8:end, 4), 'b', 'LineWidth', 2, 'DisplayName', legend_entries{5});
plot(predict_every+1:steps, SIR_RNN(8:end, 5), 'LineStyle','-.','Color', 'b', 'LineWidth', 2, 'DisplayName', legend_entries{6});
plot(predict_every+1:steps, SIR_RNN(8:end, 6), 'Color', 'g', 'LineWidth', 2, 'DisplayName', legend_entries{7});
plot(predict_every+1:steps, SIR_RNN(8:end, 7),'LineStyle','-.','Color', 'g', 'LineWidth', 2, 'DisplayName', legend_entries{8});
%}

% Update y-limits after plotting to ensure patches and lines cover full range
yLimits = ylim;

% Add vertical lines at the points where valid data starts
for col = 2:7
    valid_start = find(SIR_RNN(:, col) ~= 0, 1, 'first');
    line([valid_start valid_start], yLimits, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1, 'HandleVisibility', 'off');
end

% Loop through the NLOS_SOI list to add the transparent yellow background
for i = 1:steps
    if NLOS_SOI(i) == 1 || NLOS_SOA1(i) == 1 || NLOS_SOA2(i) == 1
        patch([i i i+1 i+1], [yLimits(1) yLimits(2) yLimits(2) yLimits(1)], ...
            	"blue", 'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end

% Add a single patch for the legend
%patch(NaN, NaN, 'blue', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', 'NLoS conditions');



% Set font size and font type (Times New Roman) for the axes
set(gca, 'FontSize', 18, 'FontName', 'Times New Roman');  % Set axes ticks font size and type

% Adding labels and title
xlabel('Step Count', 'FontName', 'Times New Roman');
ylabel('SIR (dB)', 'FontName', 'Times New Roman');
title('SIR Performance Comparison', 'FontName', 'Times New Roman');
grid on;
% Add a legend
legend('show')
legend('Location', 'southoutside', 'Orientation', 'horizontal','NumColumns',2);  % Place legend outside and below the figure


% Hold off to stop adding more plots to this figure
hold off;



%%
% Plotting
START_STEP = 1;
steps = size(SS_SV, 1);  % Assuming 'steps' is defined somewhere before

% Create a new figure
figure;
step_count = linspace(1, steps, steps);

% Plot the lists
hold on;

xlim([1, steps - START_STEP + 1]);
ylim([-58,-46]);

% Plot SS_SV and SS_RNN(:, 1) from START_STEP to the end
plot(SS_SV(START_STEP:end), 'LineStyle', '-','Marker','o','Color', '#F1D400', 'DisplayName', '(Real DoA) + Beam-steering', 'LineWidth', 2);
plot(SS_RNN(START_STEP:end, 1),'LineStyle', '--','Marker','+', 'Color','k' ,'DisplayName', '(Real DoAs) + RNN-ZF', 'LineWidth', 2);

% Plot SS_RNN(:, 2:7) starting from the first non-zero value with correct colors and vertical lines
legend_entries = {'Beam-steering', '(Real DoAs) + RNN-NS', '(TNN-small DoAs) + RNN-ZF', ...
                  '(TNN-big DoAs) + RNN-ZF', '(LSTM-MTO DoAs) + RNN-ZF', '(LSTM-MTM DoAs) + RNN-ZF', ...
                  '(GRU-MTO DoAs) + RNN-ZF', '(GRU-MTM DoAs) + RNN-ZF'};

plot(predict_every+1:steps, SS_RNN(8:end, 2),'Color','r', 'LineWidth', 2, 'DisplayName', legend_entries{3});
%plot(predict_every+1:steps, SS_RNN(8:end, 3),'LineStyle','-.', 'Color','r',  'LineWidth', 2, 'DisplayName', legend_entries{4});
%plot(predict_every+1:steps, SS_RNN(8:end, 4), 'b', 'LineWidth', 2, 'DisplayName', legend_entries{5});
%plot(predict_every+1:steps, SS_RNN(8:end, 5), 'LineStyle','-.','Color', 'b', 'LineWidth', 2, 'DisplayName', legend_entries{6});
%plot(predict_every+1:steps, SS_RNN(8:end, 6), 'Color', 'g', 'LineWidth', 2, 'DisplayName', legend_entries{7});
%plot(predict_every+1:steps, SS_RNN(8:end, 7),'LineStyle','-.','Color', 'g', 'LineWidth', 2, 'DisplayName', legend_entries{8});


% Update y-limits after plotting to ensure patches and lines cover full range
yLimits = ylim;

% Add vertical lines at the points where valid data starts
for col = 2:7
    valid_start = find(SS_RNN(:, col) ~= 0, 1, 'first');
    line([valid_start valid_start], yLimits, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1, 'HandleVisibility', 'off');
end

% Loop through the NLOS_SOI list to add the transparent yellow background
for i = 1:steps
    if NLOS_SOI(i) == 1
        patch([i i i+1 i+1], [yLimits(1) yLimits(2) yLimits(2) yLimits(1)], ...
            'green', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end

% Add a single patch for the legend
%patch(NaN, NaN, 'green', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'NLoS conditions');

% Set font size and font type (Times New Roman) for the axes
set(gca, 'FontSize', 18, 'FontName', 'Times New Roman');  % Set axes ticks font size and type

% Adding labels and title
xlabel('Step Count', 'FontName', 'Times New Roman');
ylabel('Received Power (dBm)', 'FontName', 'Times New Roman');
title('Desired User Signal Strength Comparison', 'FontName', 'Times New Roman');

% Add a legend
legend('show')
legend('Location', 'southoutside', 'Orientation', 'horizontal','NumColumns',2);  % Place legend outside and below the figure

grid on;
% Hold off to stop adding more plots to this figure
hold off;


%%
% Define START_INDEX
START_INDEX = 8;  % Start plotting from step 8

% Create a new figure for plotting angular distance errors
figure;

% Define the step range for plotting
step_range = START_INDEX:30;

% Plot Angular Distance Errors for SoI (Desired User)
% Subplot for SoI Track Prediction Error
subplot(1, 3, 1);
hold on;

% Draw a horizontal dashed line at 2 degrees to indicate an acceptable error threshold
yline(2, '--', 'LineWidth', 2, 'HandleVisibility', 'off');

% Plot the angular errors for the TNNS model
plot(step_range, SOI_errors_TNNS, 'r', 'DisplayName', 'TNNS', 'LineWidth', 2);

% Commented out plots for other models
% plot(step_range, SOI_errors_TNNB, 'LineStyle', '-.', 'Color', 'r', 'DisplayName', 'TNNB', 'LineWidth', 2);
% plot(step_range, SOI_errors_LSTM_MTO, 'b', 'DisplayName', 'MTO', 'LineWidth', 2);
% plot(step_range, SOI_errors_LSTM_MTM, 'LineStyle', '-.', 'Color', 'b', 'DisplayName', 'MTM', 'LineWidth', 2);
% plot(step_range, SOI_errors_GRU_MTO, 'g', 'DisplayName', 'GRU-MTO', 'LineWidth', 2);
% plot(step_range, SOI_errors_GRU_MTM, 'LineStyle', '-.', 'Color', 'g', 'DisplayName', 'GRU-MTM', 'LineWidth', 2);

% Add a vertical dashed line at the START_INDEX to indicate where plotting begins
xline(START_INDEX, '--k', 'LineWidth', 2);

% Get current y-axis limits for patch plotting
yLimits = ylim;

% Plot NLoS (Non-Line-of-Sight) background where NLOS_SOI is true
for i = START_INDEX:29
    if NLOS_SOI(i)
        % Add a transparent blue patch to indicate NLoS conditions
        patch([i i i+1 i+1], [yLimits(1) yLimits(2) yLimits(2) yLimits(1)], ...
            'blue', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end

% Change y-axis to logarithmic scale for better visualization of error differences
set(gca, 'YScale', 'log');

% Set custom y-tick format to fixed-point notation
ytickformat('%.1f');

% Set font size and type for the axes
set(gca, 'FontSize', 18, 'FontName', 'Times New Roman');

% Add labels and title to the subplot
xlabel('Step Count', 'FontName', 'Times New Roman');
ylabel('Angular Divergence (°)', 'FontName', 'Times New Roman');
title('SoI Track Prediction Error', 'FontName', 'Times New Roman');

hold off;

% Plot Angular Distance Errors for 1st SoA (Interference #1)
% Subplot for 1st SoA Track Prediction Error
subplot(1, 3, 2);
hold on;

% Draw a horizontal dashed line at 2 degrees
yline(2, '--', 'LineWidth', 2, 'HandleVisibility', 'off');

% Plot the angular errors for the TNNS model
plot(step_range, SOA1_errors_TNNS, 'r', 'DisplayName', 'TNNS', 'LineWidth', 2);

% Commented out plots for other models
% plot(step_range, SOA1_errors_TNNB, 'LineStyle', '-.', 'Color', 'r', 'DisplayName', 'TNNB', 'LineWidth', 2);
% plot(step_range, SOA1_errors_LSTM_MTO, 'b', 'DisplayName', 'MTO', 'LineWidth', 2);
% plot(step_range, SOA1_errors_LSTM_MTM, 'LineStyle', '-.', 'Color', 'b', 'DisplayName', 'MTM', 'LineWidth', 2);
% plot(step_range, SOA1_errors_GRU_MTO, 'g', 'DisplayName', 'GRU-MTO', 'LineWidth', 2);
% plot(step_range, SOA1_errors_GRU_MTM, 'LineStyle', '-.', 'Color', 'g', 'DisplayName', 'GRU-MTM', 'LineWidth', 2);

% Add a vertical dashed line at the START_INDEX
xline(START_INDEX, '--k', 'LineWidth', 2);

% Get current y-axis limits
yLimits = ylim;

% Plot NLoS background where NLOS_SOA1 is true
for i = START_INDEX:29
    if NLOS_SOA1(i)
        % Add a transparent blue patch for NLoS conditions
        patch([i i i+1 i+1], [yLimits(1) yLimits(2) yLimits(2) yLimits(1)], ...
            'blue', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end

% Change y-axis to logarithmic scale
set(gca, 'YScale', 'log');

% Set custom y-tick format
ytickformat('%.1f');

% Set font size and type for the axes
set(gca, 'FontSize', 18, 'FontName', 'Times New Roman');

% Add labels and title to the subplot
xlabel('Step Count', 'FontName', 'Times New Roman');
ylabel('Angular Divergence (°)', 'FontName', 'Times New Roman');
title('1st SoA Track Prediction Error', 'FontName', 'Times New Roman');

hold off;

% Plot Angular Distance Errors for 2nd SoA (Interference #2)
% Subplot for 2nd SoA Track Prediction Error
subplot(1, 3, 3);
hold on;

% Draw a horizontal dashed line at 2 degrees
yline(2, '--', 'LineWidth', 2, 'HandleVisibility', 'off');

% Plot the angular errors for the TNNS model
plot(step_range, SOA2_errors_TNNS, 'r', 'DisplayName', 'TNNS', 'LineWidth', 2);

% Commented out plots for other models
% plot(step_range, SOA2_errors_TNNB, 'LineStyle', '-.', 'Color', 'r', 'DisplayName', 'TNNB', 'LineWidth', 2);
% plot(step_range, SOA2_errors_LSTM_MTO, 'b', 'DisplayName', 'MTO', 'LineWidth', 2);
% plot(step_range, SOA2_errors_LSTM_MTM, 'LineStyle', '-.', 'Color', 'b', 'DisplayName', 'MTM', 'LineWidth', 2);
% plot(step_range, SOA2_errors_GRU_MTO, 'g', 'DisplayName', 'GRU-MTO', 'LineWidth', 2);
% plot(step_range, SOA2_errors_GRU_MTM, 'LineStyle', '-.', 'Color', 'g', 'DisplayName', 'GRU-MTM', 'LineWidth', 2);

% Add the vertical dashed line at x = START_INDEX
xline(START_INDEX, '--k', 'LineWidth', 2);

% Add a single handle for the legend (since only one model is plotted)
h1 = plot(step_range, SOA2_errors_TNNS, 'r', 'DisplayName', 'TNN-small', 'LineWidth', 2);

% Commented out handles for other models
% h2 = plot(step_range, SOA2_errors_TNNB, 'LineStyle', '-.', 'Color', 'r', 'DisplayName', 'TNN-big', 'LineWidth', 2);
% h3 = plot(step_range, SOA2_errors_LSTM_MTO, 'b', 'DisplayName', 'LSTM-MTO', 'LineWidth', 2);
% h4 = plot(step_range, SOA2_errors_LSTM_MTM, 'LineStyle', '-.', 'Color', 'b', 'DisplayName', 'LSTM-MTM', 'LineWidth', 2);
% h5 = plot(step_range, SOA2_errors_GRU_MTO, 'g', 'DisplayName', 'GRU-MTO', 'LineWidth', 2);
% h6 = plot(step_range, SOA2_errors_GRU_MTM, 'LineStyle', '-.', 'Color', 'g', 'DisplayName', 'GRU-MTM', 'LineWidth', 2);

% Add the legend to the subplot
hLegend = legend([h1], 'Location', 'southoutside', 'Orientation', 'horizontal');

% Optionally, adjust the legend's position if needed
% pos = get(hLegend, 'Position');
% pos(2) = 0.01;  % Lower the legend to the bottom of the figure
% set(hLegend, 'Position', pos);

% Change y-axis to logarithmic scale
set(gca, 'YScale', 'log');
ytickformat('%.1f');  % Display tick labels in fixed-point notation

% Get current y-axis limits
yLimits = ylim;

% Plot NLoS background where NLOS_SOA2 is true
for i = START_INDEX:29
    if NLOS_SOA2(i)
        % Add a transparent blue patch for NLoS conditions
        patch([i i i+1 i+1], [yLimits(1) yLimits(2) yLimits(2) yLimits(1)], ...
            'blue', 'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end

% Set y-axis limits for consistency across subplots (optional)
ylim([0, 26.6]);

% Set font size and type for the axes
set(gca, 'FontSize', 18, 'FontName', 'Times New Roman');

% Add labels and title to the subplot
xlabel('Step Count', 'FontName', 'Times New Roman');
ylabel('Angular Divergence (°)', 'FontName', 'Times New Roman');
title('2nd SoA Track Prediction Error', 'FontName', 'Times New Roman');

hold off;

%% Visualization of Transmitter and Receivers on the Map
% Clear the current map in the site viewer
clearMap(viewer);

% Show the transmitter site on the map
show(tx);

% Loop through each step to display the receivers and perform ray tracing
for step = 1:steps

    disp("Step " + step + ":");

    % Define the desired receiver (SoI)
    rx = rxsite("Latitude", track_SOI(step, 1), "Longitude", track_SOI(step, 2), ...
        "AntennaHeight", 2);

    % Define the interfering receivers (SoA1 and SoA2)
    rx1 = rxsite("Latitude", track_SOA1(step, 1), "Longitude", track_SOA1(step, 2), ...
        "AntennaHeight", 2);
    rx2 = rxsite("Latitude", track_SOA2(step, 1), "Longitude", track_SOA2(step, 2), ...
        "AntennaHeight", 2);

    % Check if the receiver is on the roof
    if rx.elevation > 26
        disp("On roof");
    end

    % Determine the appropriate propagation model
    class = 1;
    [rtpm, noRay] = properProp(class);

    % Perform ray tracing between transmitter and receivers
    ray = raytrace(tx, rx, rtpm);
    SoA1_ray = raytrace(tx, rx1, rtpm);
    SoA2_ray = raytrace(tx, rx2, rtpm);

    % Visualize the receivers and rays every other step
    if mod(step, 2) == 0
        % Show the receivers with custom icons
        show(rx, "Icon", "desired.png");
        show(rx1, "Icon", "interference1.png");
        show(rx2, "Icon", "interference2.png");

        % Perform and display ray tracing on the map
        raytrace(tx, rx, rtpm);
        raytrace(tx, rx1, rtpm);
        raytrace(tx, rx2, rtpm);
    end

end

% Display the antenna pattern of the transmitter with full transparency
pattern(tx, "Transparency", 1);



%% Function to calculate angular distance in degrees
function theta = angular_distance_deg(az1, el1, az2, el2)
    theta = acosd(sind(el1) .* sind(el2) + cosd(el1) .* cosd(el2) .* cosd(az1 - az2));
end