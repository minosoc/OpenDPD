%__author__ = "Yizhuo Wu, Chang Gao"
%__license__ = "Apache-2.0 License"
%__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"
%
% Script to create unnormalized I/Q dataset from instrument measurements
% This script downloads I/Q data from instrument and saves it directly
% without normalization

function create_dataset_from_instrument(output_dir, dataset_name)
    % Parameters
    %   output_dir: Directory to save the dataset (e.g., '../datasets')
    %   dataset_name: Name of the dataset folder (e.g., 'APA_200MHz')
    
    % Example measurement parameters - adjust these for your setup
    Ts = 1000;              % Time length in us
    fc = 3.5e9;            % Center frequency in Hz
    fs = 983.04e6;         % Sampling frequency in Hz
    Ns = 98304;            % Number of samples
    
    % Create output directory
    dataset_path = fullfile(output_dir, dataset_name);
    if ~exist(dataset_path, 'dir')
        mkdir(dataset_path);
    end
    
    % Download input signal (before PA) - adjust this based on your setup
    fprintf('Downloading input I/Q data...\n');
    input_data = N9042B_IQdownload(Ts, fc, fs, Ns);
    I_in = real(input_data);
    Q_in = imag(input_data);
    
    % Download output signal (after PA) - adjust this based on your setup
    fprintf('Downloading output I/Q data...\n');
    output_data = N9042B_IQdownload(Ts, fc, fs, Ns);
    I_out = real(output_data);
    Q_out = imag(output_data);
    
    % Create table with unnormalized data
    data_table = table(I_in, Q_in, I_out, Q_out, ...
                      'VariableNames', {'I_in', 'Q_in', 'I_out', 'Q_out'});
    
    % Save to CSV (unnormalized)
    csv_path = fullfile(dataset_path, 'data.csv');
    writetable(data_table, csv_path);
    fprintf('Unnormalized dataset saved to: %s\n', csv_path);
    
    % Display statistics
    fprintf('\nData Statistics:\n');
    fprintf('Input I range: [%.6f, %.6f]\n', min(I_in), max(I_in));
    fprintf('Input Q range: [%.6f, %.6f]\n', min(Q_in), max(Q_in));
    fprintf('Input magnitude range: [%.6f, %.6f]\n', ...
            min(sqrt(I_in.^2 + Q_in.^2)), max(sqrt(I_in.^2 + Q_in.^2)));
    fprintf('Output I range: [%.6f, %.6f]\n', min(I_out), max(I_out));
    fprintf('Output Q range: [%.6f, %.6f]\n', min(Q_out), max(Q_out));
    fprintf('Output magnitude range: [%.6f, %.6f]\n', ...
            min(sqrt(I_out.^2 + Q_out.^2)), max(sqrt(I_out.^2 + Q_out.^2)));
    
    % Calculate gain ratio
    input_mag = sqrt(I_in.^2 + Q_in.^2);
    output_mag = sqrt(I_out.^2 + Q_out.^2);
    valid_idx = input_mag > 0.01;
    if any(valid_idx)
        gain_ratio = mean(output_mag(valid_idx) ./ input_mag(valid_idx));
        fprintf('Average gain ratio (output/input): %.6f\n', gain_ratio);
    end
    
    % Create spec.json with normalization_scale information
    % Note: If your data is already normalized, set normalization_scale to 1.0
    % Otherwise, set it to the original scale factor used for normalization
    spec = struct();
    spec.dataset_format = 'single_csv';
    spec.split_ratios = struct('train', 0.6, 'val', 0.2, 'test', 0.2);
    spec.input_signal_fs = fs;
    spec.nperseg = 19662;  % Adjust based on your needs
    spec.gain = gain_ratio;  % Use calculated gain ratio
    spec.normalization_scale = 1.0;  % Set to 1.0 if data is unnormalized
    
    spec_path = fullfile(dataset_path, 'spec.json');
    json_str = jsonencode(spec, 'PrettyPrint', true);
    fid = fopen(spec_path, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    fprintf('spec.json saved to: %s\n', spec_path);
end

