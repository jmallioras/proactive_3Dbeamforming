% In this script we invoke the python scripts that load the pre-trained models and given the input DoA trajectories return
% the future DoA.

function [pred_SOI, pred_SOA1, pred_SOA2] = proactive_models(currentPath_SOI, current_path_SOA1, current_path_SOA2)
    pyenv('Version', "/Users/ioannis.mallioras/anaconda3/bin/python");
    pred_SOI = zeros(6,2); % MODEL / [AZ,EL]
    pred_SOA1 = zeros(6,2);
    pred_SOA2 = zeros(6,2);
    
    % =============== TNN PROACTIVE ============== %
    tnn_input = [currentPath_SOI, current_path_SOA1, current_path_SOA2];
    res = pyrunfile("/model_inference/run_tnn.py","output",x=tnn_input);
    
    % Convert Python lists to MATLAB cells, then to double arrays
    doa_tnn_small_SOI = cellfun(@double, cell(res(1)),'UniformOutput',false);
    doa_tnn_small_SOA1 = cellfun(@double, cell(res(2)),'UniformOutput',false);
    doa_tnn_small_SOA2 = cellfun(@double, cell(res(3)),'UniformOutput',false);

    % ACTIVATE COMMENTED LINES IF A BIG TNN MODEL IS AVAILABLE
    %{
    doa_tnn_big_SOI = cellfun(@double, cell(res(4)),'UniformOutput',false);
    doa_tnn_big_SOA1 = cellfun(@double, cell(res(5)),'UniformOutput',false);
    doa_tnn_big_SOA2 = cellfun(@double, cell(res(6)),'UniformOutput',false);
    %}
    pred_SOI(1,:) = doa_tnn_small_SOI{1,1};
    %pred_SOI(2,:) = doa_tnn_big_SOI{1,1};

    pred_SOA1(1,:) = doa_tnn_small_SOA1{1,1};
    %pred_SOA1(2,:) = doa_tnn_big_SOA1{1,1};

    pred_SOA2(1,:) = doa_tnn_small_SOA2{1,1};
    %pred_SOA2(2,:) = doa_tnn_big_SOA2{1,1};


    % ACTIVATE IF LSTM/GRU (MTM AND/OR MTO) MODELS ARE AVAILABLE

    %{

    % =============== LSTM PROACTIVE ============== %
    rnn_input = [currentPath_SOI, current_path_SOA1, current_path_SOA2];
    res = pyrunfile("/model_inference/run_lstm.py","output",x=rnn_input);
    % Convert Python lists to MATLAB cells, then to double arrays
    doa_lstm_mto_SOI = cellfun(@double, cell(res(1)),'UniformOutput',false);
    doa_lstm_mto_SOA1 = cellfun(@double, cell(res(2)),'UniformOutput',false);
    doa_lstm_mto_SOA2 = cellfun(@double, cell(res(3)),'UniformOutput',false);

    doa_lstm_mtm_SOI = cellfun(@double, cell(res(4)),'UniformOutput',false);
    doa_lstm_mtm_SOA1 = cellfun(@double, cell(res(5)),'UniformOutput',false);
    doa_lstm_mtm_SOA2 = cellfun(@double, cell(res(6)),'UniformOutput',false);

    pred_SOI(3,:) = doa_lstm_mto_SOI{1,1};
    pred_SOI(4,:) = doa_lstm_mtm_SOI{1,1};

    pred_SOA1(3,:) = doa_lstm_mto_SOA1{1,1};
    pred_SOA1(4,:) = doa_lstm_mtm_SOA1{1,1};

    pred_SOA2(3,:) = doa_lstm_mto_SOA2{1,1};
    pred_SOA2(4,:) = doa_lstm_mtm_SOA2{1,1};

    % =============== GRU PROACTIVE ============== %
    gru_input = [currentPath_SOI, current_path_SOA1, current_path_SOA2];
    res = pyrunfile("/model_inference/run_gru.py","output",x=gru_input);
    % Convert Python lists to MATLAB cells, then to double arrays
    doa_gru_mto_SOI = cellfun(@double, cell(res(1)),'UniformOutput',false);
    doa_gru_mto_SOA1 = cellfun(@double, cell(res(2)),'UniformOutput',false);
    doa_gru_mto_SOA2 = cellfun(@double, cell(res(3)),'UniformOutput',false);

    doa_gru_mtm_SOI = cellfun(@double, cell(res(4)),'UniformOutput',false);
    doa_gru_mtm_SOA1 = cellfun(@double, cell(res(5)),'UniformOutput',false);
    doa_gru_mtm_SOA2 = cellfun(@double, cell(res(6)),'UniformOutput',false);

    pred_SOI(5,:) = doa_gru_mto_SOI{1,1};
    pred_SOI(6,:) = doa_gru_mtm_SOI{1,1};

    pred_SOA1(5,:) = doa_gru_mto_SOA1{1,1};
    pred_SOA1(6,:) = doa_gru_mtm_SOA1{1,1};

    pred_SOA2(5,:) = doa_gru_mto_SOA2{1,1};
    pred_SOA2(6,:) = doa_gru_mtm_SOA2{1,1};
    %}
end
