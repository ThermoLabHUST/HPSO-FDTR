% Main Function
clc
clear all
tic
%Input Control Parameters
kz=[160 0.1   120    10    80   130]; %W/m-K
kr=[160 0.1   120    10    80   130];
C=[2.44 0.01  2.6    2.6   2.4  1.665]; %J/m3-K
t=[87.4 1    1.08e3 0.46e3 290 290e3]*1e-9; %m
A_pump=1e-3; %laser power (Watts) . . . only used for amplitude est.
TCR=1; %coefficient of thermal reflectance . . . only used for amplitude est.

mult_Auto_fit=1;

 if mult_Auto_fit==1
%Configure parameters for fitting: 1 indicates the parameter should be fitted, while 0 means it should remain fixed
xu_flags = [0 1 1 1 0 1 ;  % kz
            0 0 0 0 0 0;  % kr
            0 0 1 0 0 0;  % C 
            0 0 0 0 0 0]; % t
 isotropic_layers = [1 1 1 1 1 1];  %Setting it to 1 indicates that the layer is isotropic, while 0 denotes anisotropy
FDTR_mult_Autofit_x(kz, kr, C, t, A_pump, TCR, xu_flags, isotropic_layers);
 end
%----------------------------------------------------
fprintf('Program Completed\n')
toc
 

% Subfunction ‘FDTR_mult_Autofit_x’ to parameter optimization
 function FDTR_mult_Autofit_x(kz, kr, C, t, A_pump, TCR, xu_flags, isotropic_layers)
 % Set the path of the Data process folder
dataFolderPath = fullfile('FDTR exp_data');

files = dir(fullfile(dataFolderPath, '*.txt'));
% Retrieve information from all txt files in the Data_process folder, where the files are sequentially numbered to facilitate subsequent matching with the spot size r settings
 fileNames = {files.name};
    prefixNumbers = zeros(1, length(fileNames));
    for i = 1:length(fileNames)
        match = regexp(fileNames{i}, '^(\d+)', 'match');
        if ~isempty(match)
            prefixNumbers(i) = str2double(match{1});
        else
            error('Filename "%s" does not start with a numeric prefix!', fileNames{i});
        end
    end
    [~, sortedIdx] = sort(prefixNumbers);
    files = files(sortedIdx);
    ccc = length(files);

% Define the r values in one-to-one correspondence with the filenames
r_values = [7.4e-6, 3.4e-6];  % Array of r values

all_params_initial = [kz, kr, C, t];  
xu_flags_transposed = xu_flags';
params_to_fit_index = find(xu_flags_transposed == 1);  
params_initial = all_params_initial(params_to_fit_index);

% Set parameter fitting ranges: either default to the range around initial control parameters, or specify a custom search range
% params_lb = params_initial / 2;
% params_ub = params_initial * 1.5;
params_lb = [0.01, 1, 1,  1,0.5];
params_ub = [3, 1000, 500, 1000,5];

% Set the objective function threshold
fitness_threshold = 10;

% Define PSO options
options = optimoptions('particleswarm', ...
    'SwarmSize', 200, ...             % Particle swarm size
    'MaxIterations', 20, ...         % Maximum number of iterations
    'InitialSwarmMatrix', params_lb + (params_ub - params_lb) .* rand(200, length(params_initial)), ... %Randomly select initial values
    'Display', 'iter', ...            % Display iteration information
    'FunctionTolerance', 1e-9, ...  % Objective function tolerance
    'OutputFcn', @(optimValues, state) stopPSOIfThresholdReached(optimValues, state, fitness_threshold)); 

% Invoke particle swarm optimization algorithm
    [best_param_fit_pso, best_fitness_pso, ~, output] = particleswarm(@(p) ...
    computeCombinedResiduals(p, params_to_fit_index, all_params_initial, dataFolderPath, files,ccc, TCR, A_pump, isotropic_layers, r_values), ...
    length(params_initial), params_lb, params_ub, options);

% Determine whether to switch to the quasi-Newton method
    if best_fitness_pso <= fitness_threshold || output.iterations >= options.MaxIterations
        fprintf(' Switch to the quasi-Newton method for further refinement...\n');
    
        fprintf('\nPSO optimization results:\n');
        for i = 1:length(best_param_fit_pso)
            fprintf('Parameter %d fit value: %f\n', i, best_param_fit_pso(i));
        end
        fprintf(' Objective function value: %f\n', best_fitness_pso);
        % Configure quasi-Newton optimization options
        options_nlbfgs = optimoptions('fminunc', ...
            'Algorithm', 'quasi-newton', ...
            'Display', 'iter', ...
            'OptimalityTolerance', 1e-9, ...
            'StepTolerance', 1e-9);
    
        % Invoke quasi-Newton optimization
            [best_param_fit_nlbfgs, best_fitness_nlbfgs] = fminunc(@(p) ...
            computeCombinedResiduals(p, params_to_fit_index, all_params_initial, dataFolderPath, files,ccc, TCR, A_pump, isotropic_layers, r_values), ...
            best_param_fit_pso, options_nlbfgs);
    
        % Display quasi-Newton results
        fprintf('\n Quasi-Newton optimization results:\n');
        for i = 1:length(best_param_fit_nlbfgs)
            fprintf(' Parameter %d fit value: %f\n', i, best_param_fit_nlbfgs(i));
        end
        fprintf(' Objective function value: %f\n', best_fitness_nlbfgs);
    else
        % Display PSO results
        fprintf('\nPSO optimization results:\n');
        for i = 1:length(best_param_fit_pso)
            fprintf(' Parameter %d fit value: %f\n', i, best_param_fit_pso(i));
        end
        fprintf(' Objective function value: %f\n', best_fitness_pso);
    end

 end
% Termination condition function
function stop = stopPSOIfThresholdReached(optimValues, ~, fitness_threshold)
    stop = false;
    if optimValues.bestfval <= fitness_threshold
        fprintf(' Objective function threshold reached - terminating PSO optimization...\n');
        stop = true;
    end
end

% Residual calculation function definition (combining residuals from multiple signals)
function F = computeCombinedResiduals(p, idx_to_fit, all_params, dataFolderPath, files,ccc, TCR, A_pump, isotropic_layers, r_values)
    % Update optimization parameters
    all_params(idx_to_fit) = p;
    NNN = (length(all_params)) / 4;
    kz_updated = all_params(1:NNN);
    kr_updated = all_params(NNN+1:2*NNN);
    C_updated = all_params(2*NNN+1:3*NNN);
    t_updated = all_params(3*NNN+1:4*NNN);

    % Determine isotropy for each layer
    for j = 1:NNN
        if isotropic_layers(j) == 1
            kr_updated(j) = kz_updated(j); 
        end
    end

    F = 0;

    % Compute residuals for each signal group
    for iii = 1:ccc
        % Construct file paths and read experimental data
        filePath = fullfile(dataFolderPath, files(iii).name);
        data = load(filePath);
        f_list = data(:, 1);  % Frequency variable
        R_exp = data(:, 2);   % Experimental phase signal

        % Retrieve the corresponding r-value
        r = r_values(iii);  % Retrieve the corresponding r-value based on filename indexing

        % Compute simulated signals
        [~, ~, sim_R] = FDTR_REFL(f_list, TCR, kz_updated, kr_updated, C_updated, t_updated, r, A_pump);

        F = F + sum((sim_R - R_exp).^2);
    end
end


% Subfunction ‘FDTR_REFL’ to calculate FDTR signals 
 function [Anorm,R,FAI] = FDTR_REFL(f_list,TCR,kz,kr,C,t,r,A_pump)

    kmax = 5/r;      % Maximum k vector to consider in integration for DeltaT
    %use Legendre-Gauss Integration, import weights and node locations...
    load GLnodes_n64.txt
    weight_0 = GLnodes_n64(:,2);
    xvect_0 = GLnodes_n64(:,3);
    weights = kmax/2*weight_0;        % (Column vector, k*1)
    kvect = kmax/2*xvect_0+kmax/2;    % k vector for integration of DeltaT (Column vector, k*1)

    DeltaTmat = FDTR_TEMP(kvect,f_list,kz,kr,C,t,r,A_pump); % (k*m)
    DeltaT = weights'*DeltaTmat;    % (1*m)

    Z = TCR*DeltaT';
    X = real(Z);
    Y = -imag(Z);
    R = -X./Y;
    FAI = atan(Y./X)/pi*180;
    
    Anorm = abs(Z)/abs(Z(1));
 end

% Subfunction ‘FDTR_TEMP’ to calculate Green's function 
function [Integrand, G] = FDTR_TEMP(kvectin, freq, kz, kr, C, t, r, A_pump)
    % This version assumes both the heat deposition and temperature detection
    % happen on the surface of the first layer
    C=C*1E6;
    Nfreq = length(freq);
    kvect = kvectin(:) * ones(1, Nfreq); % Ensure kvect is a matrix with size [Nk, Nfreq]
    Nlayers = length(kz); % Number of layers
    Nk = length(kvectin); % Number of different frequencies to calculate for

    % Ensure freq is a row vector
    freq = freq(:)';
    
    % alpha and eta should be column vectors
    alpha = kz ./ C;
    eta = kr ./ kz;
    
    omega = 2 * pi * freq;
    
    % Ensure q2 and other vectors are of correct dimensions
    q2 = (1i * omega) ./ alpha(Nlayers); % q2 should be a row vector
    q2 = repmat(q2, Nk, 1); % Replicate q2 to match kvect's dimensions
    
    kvect2 = kvect.^2;
    
    un = sqrt(4 * pi^2 * eta(Nlayers) * kvect2 + q2);
    gamman = kz(Nlayers) * un;
    
    Bplus = zeros(Nk, Nfreq);
    Bminus = ones(Nk, Nfreq);
    kterm2 = 4 * pi^2 * kvect2;
    
    if Nlayers ~= 1
        for n = Nlayers:-1:2
            q2 = (1i * omega) ./ alpha(n-1);
            q2 = repmat(q2, Nk, 1); % Replicate q2 to match kvect's dimensions
            
            unminus = sqrt(eta(n-1) * kterm2 + q2);
            gammanminus = kz(n-1) * unminus;
            
            AA = gammanminus + gamman;
            BB = gammanminus - gamman;
            
            temp1 = AA .* Bplus + BB .* Bminus;
            temp2 = BB .* Bplus + AA .* Bminus;
            
            expterm = exp(unminus * t(n-1));
            
            Bplus = (0.5 ./ (gammanminus .* expterm)) .* temp1;
            Bminus = (0.5 ./ gammanminus) .* expterm .* temp2;
            
            % Numerical stability fix
            penetration_logic = logical(t(n-1) * abs(unminus) > 100); % If penetration is smaller than layer...set to semi-inf
            Bplus(penetration_logic) = 0;
            Bminus(penetration_logic) = 1;
            
            un = unminus;
            gamman = gammanminus;
        end
    end

    G = (Bplus + Bminus) ./ (Bminus - Bplus) ./ gamman; % The layer G(k)
    Kernal = 2 * pi * A_pump * exp(-pi^2 * r^2 * kvect.^2) .* kvect; % The rest of the integrand
    Integrand = G .* Kernal;
 end

