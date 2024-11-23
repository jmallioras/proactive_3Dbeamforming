% This function invokes the python script that loads the pre-trained zero-forcing LSTM beamformer presented in https://ieeexplore.ieee.org/document/10438855
% This model takes as input three DoAs (first is the desired signal DoA and the latter two are the undesired ones) and
% returns the beamforming weight vector for an 8x8 planar array (64 real and 64 imaginary values).

function nn_sv = RNN(nn_input)
    pyenv('Version', "/Users/ioannis.mallioras/anaconda3/bin/python");
    th_max = 59.9;
    th_min = 0.0;
    ph_max = 359.9;
    ph_min = 0.0;
    
    % Clip the angles the RNN does not support
    nn_input(1:3) = max(min(nn_input(1:3),th_max),th_min);
    nn_input(4:6) = max(min(nn_input(4:6),ph_max),ph_min);

    norm_input = [norm(nn_input(1:3),th_min,th_max),norm(nn_input(4:6),ph_min,ph_max)];
    
    res = pyrunfile("/model_inference/run_rnn.py","output",x=norm_input);
    
    result = cellfun(@double,cell(res)); %pylist-> cell array -> double array
    nn_sv = complex(result(1:64),result(65:128));
    nn_sv = reshape(nn_sv, [64,1]);
end

function y = norm(x, x_min, x_max)
    y = (x - x_min) / (x_max - x_min);
end