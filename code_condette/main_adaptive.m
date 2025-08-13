% -- CUSTOM SET UP --
gridsize = 1;
N = 512;                   % Number of Gridpoints
dt = 1/5000;               % Initial time step
dt_min = 1e-15;            % Minimum allowed time step
dt_max = 1e-2;             % Maximum allowed time step
c0 = 9/32;                 % Normalization constant for double well

epsilon = 1/20;
gamma = 1/400;             % Paper parameter

th = 1;                    % Thickness parameter
Nmax = 400;                 % Max iterations for fixed point
tol = 10^(-6);              % Tolerance for fixed point convergence

stop_crit = 10^(-10);       % Stopping criterion for time iteration
max_it = 200000;

% ADAPTIVE TIME STEPPING PARAMETERS
target_fp_iterations = 10;  % Target number of fixed point iterations
max_fp_iterations = 50;     % Maximum allowed fixed point iterations
min_fp_iterations = 3;      % Minimum required fixed point iterations
dt_increase_factor = 1.2;   % Factor to increase dt when converging quickly
dt_decrease_factor = 0.5;   % Factor to decrease dt when converging slowly
energy_change_threshold = 1e-8;  % Threshold for energy change control
max_solution_change = 0.1;  % Maximum allowed solution change per step

% -- GENERIC SET UP --
x = gridsize/N * (0:N-1);
k = [0:N/2-1 -N/2:-1];     % Proper wave number vector for FFT

[xi, eta] = ndgrid(k, k);   % 2D wave numbers
modk2 = xi.^2 + eta.^2;
modk = sqrt(modk2);

% IMPROVED SIGMA FUNCTION
function sig = sigma(A)
    sig = zeros(size(A));
    zero_freq = (abs(A) < 1e-14);
    small = (abs(A) >= 1e-14) & (abs(A) < 1e-6);
    large = (abs(A) >= 1e-6);
    
    sig(zero_freq) = 1;
    sig(small) = 1 - pi * abs(A(small));
    sig(large) = (1 - exp(-2 * pi * abs(A(large)))) ./ (2 * pi * abs(A(large)));
end

function [k, u_n, err, conv] = fixpt(u0, L, dt, N, epsilon, gamma, Nmax, tol, c0)
    % Nonlinear function
    NL = @(u, v, epsilon, gamma, c0) 2 * gamma * c0 / epsilon * (u + v) .* (1 - (abs(u).^2 + abs(v).^2) / 2);

    % Precompute FFT of initial condition
    fft_u0 = fft2(u0);
    
    % Constant term for iteration
    denominator = ones(N) + dt/2 * L;
    numerator = ones(N) - dt/2 * L;
    
    % Avoid division by zero
    denominator(denominator == 0) = 1e-14;
    
    CT = real(ifft2(numerator ./ denominator .* fft_u0));

    % Initialization
    j = 1;
    error = 10;
    u_int = u0;
    u_int2 = zeros(N,N);
    convergence = 0;

    % Fixed Point Iteration Loop
    while (j <= Nmax) && (error > tol)
        nl_term = NL(u_int, u0, epsilon, gamma, c0);
        
        fft_nl = fft2(nl_term);
        u_int2 = real(ifft2(dt * fft_nl ./ denominator) + CT);
        
        error = max(max(abs(u_int2 - u_int)));
        
        % Relaxation for better convergence
        alpha = 0.7;
        u_int = alpha * u_int2 + (1 - alpha) * u_int;
        j = j + 1;
    end

    if error <= tol
        convergence = 1;
    end

    k = j;
    u_n = u_int2;
    err = error;
    conv = convergence;
end

function [discrete_energy_value] = energy_value(gamma, epsilon, N, u, th, modk, modk2, c0)
    W = double_well(u, c0);
    ftu = fft2(u) / N;
    
    S = sigma(th * modk);
    
    kinetic_energy = 0.5 * sum(sum((S + gamma * epsilon * modk2) .* abs(ftu).^2));
    potential_energy = (gamma / epsilon) * sum(sum(W)) / N^2;
    
    discrete_energy_value = potential_energy + kinetic_energy;
end

function W = double_well(u, c0)
    u2 = 1 - abs(u).^2;
    W = c0 * (u2.^2);
end

% ADAPTIVE TIME STEPPING FUNCTIONS
function [new_dt, accept_step] = adapt_timestep_convergence(dt, fp_iterations, target_fp_iterations, ...
    max_fp_iterations, min_fp_iterations, dt_increase_factor, dt_decrease_factor, dt_min, dt_max)
    
    accept_step = true;
    new_dt = dt;
    
    if fp_iterations > max_fp_iterations
        % Fixed point didn't converge - reject step and decrease dt
        new_dt = max(dt * dt_decrease_factor, dt_min);
        accept_step = false;
        fprintf('Step rejected: FP iterations = %d, reducing dt to %.2e\n', fp_iterations, new_dt);
    elseif fp_iterations < min_fp_iterations
        % Converged too quickly - increase dt
        new_dt = min(dt * dt_increase_factor, dt_max);
        fprintf('Fast convergence: FP iterations = %d, increasing dt to %.2e\n', fp_iterations, new_dt);
    elseif fp_iterations > target_fp_iterations * 1.5
        % Converging slowly - decrease dt
        new_dt = max(dt * dt_decrease_factor, dt_min);
        fprintf('Slow convergence: FP iterations = %d, decreasing dt to %.2e\n', fp_iterations, new_dt);
    elseif fp_iterations < target_fp_iterations * 0.5
        % Converging quickly - can increase dt
        new_dt = min(dt * dt_increase_factor, dt_max);
        fprintf('Quick convergence: FP iterations = %d, increasing dt to %.2e\n', fp_iterations, new_dt);
    end
end

function [new_dt, accept_step] = adapt_timestep_energy(dt, energy_change, energy_change_threshold, ...
    dt_decrease_factor, dt_increase_factor, dt_min, dt_max)
    
    accept_step = true;
    new_dt = dt;
    
    if energy_change > energy_change_threshold * 10
        % Energy change too large - decrease dt
        new_dt = max(dt * dt_decrease_factor, dt_min);
        accept_step = false;
        fprintf('Step rejected: Energy change = %.2e, reducing dt to %.2e\n', energy_change, new_dt);
    elseif energy_change < energy_change_threshold * 0.1
        % Energy change very small - can increase dt
        new_dt = min(dt * dt_increase_factor, dt_max);
    end
end

function [new_dt, accept_step] = adapt_timestep_solution_change(dt, solution_change, max_solution_change, ...
    dt_decrease_factor, dt_increase_factor, dt_min, dt_max)
    
    accept_step = true;
    new_dt = dt;
    
    if solution_change > max_solution_change
        % Solution change too large - decrease dt
        new_dt = max(dt * dt_decrease_factor, dt_min);
        accept_step = false;
        fprintf('Step rejected: Solution change = %.2e, reducing dt to %.2e\n', solution_change, new_dt);
    elseif solution_change < max_solution_change * 0.1
        % Solution change small - can increase dt
        new_dt = min(dt * dt_increase_factor, dt_max);
    end
end

% IMPROVED INITIAL CONDITION
amplitude = 0.1;
u0 = amplitude * (2*rand(N,N) - 1) + 1i * amplitude * (2*rand(N,N) - 1);

[x1, x2] = ndgrid(x, x);
perturbation = 0.01 * sin(2*pi*x1) .* cos(2*pi*x2);
u0 = u0 + perturbation;

% LINEAR OPERATOR
L = gamma * epsilon * modk2 + sigma(th * modk);
L(1,1) = sigma(0);

% -- TIME STEPPING LOOP INITIALIZATION --
n_it = 1;
time = 0;
i = 1;
u_int = u0;
u_prev = u0;  % For solution change calculation
Denergy = 1000;
consecutive_rejections = 0;
max_consecutive_rejections = 10;

% -- TIME VECTOR VALUES --
time_vector = zeros(max_it, 1);
time_vector(1) = 0;
dt_history = zeros(max_it, 1);
dt_history(1) = dt;

% -- ENERGY COMPUTATION - INITIALIZATION --
Energy = zeros(1, max_it);
Energy(1,1) = energy_value(gamma, epsilon, N, u0, th, modk, modk2, c0);
i = 2;

fig1 = figure(1);
fig2 = figure(2);
fig3 = figure(3);

fprintf('Starting adaptive time stepping simulation...\n');
fprintf('Initial dt = %.2e\n', dt);

% -- ADAPTIVE TIME STEPPING LOOP --
while (n_it < max_it) && (abs(Denergy) > stop_crit) && (dt >= dt_min)
    
    % Attempt time step
    [k_fp, u_new, err, conv] = fixpt(u_int, L, dt, N, epsilon, gamma, Nmax, tol, c0);
    
    accept_step = false;
    
    if conv == 1
        % Calculate metrics for adaptive control
        current_energy = energy_value(gamma, epsilon, N, u_new, th, modk, modk2, c0);
        
        if i > 1
            energy_change = abs(current_energy - Energy(1,i-1));
            solution_change = max(max(abs(u_new - u_int)));
            
            % Apply multiple adaptive criteria
            [dt1, accept1] = adapt_timestep_convergence(dt, k_fp, target_fp_iterations, ...
                max_fp_iterations, min_fp_iterations, dt_increase_factor, dt_decrease_factor, dt_min, dt_max);
            
            [dt2, accept2] = adapt_timestep_energy(dt, energy_change, energy_change_threshold, ...
                dt_decrease_factor, dt_increase_factor, dt_min, dt_max);
            
            [dt3, accept3] = adapt_timestep_solution_change(dt, solution_change, max_solution_change, ...
                dt_decrease_factor, dt_increase_factor, dt_min, dt_max);
            
            % Accept step only if all criteria are satisfied
            accept_step = accept1 && accept2 && accept3;
            
            % Choose the most conservative dt
            new_dt = min([dt1, dt2, dt3]);
        else
            % First step - just use convergence criterion
            [new_dt, accept_step] = adapt_timestep_convergence(dt, k_fp, target_fp_iterations, ...
                max_fp_iterations, min_fp_iterations, dt_increase_factor, dt_decrease_factor, dt_min, dt_max);
        end
        
        if accept_step
            % Step accepted - update solution
            Energy(1,i) = current_energy;
            Denergy = abs(Energy(1,i-1) - Energy(1,i));
            
            time_vector(i) = time_vector(i-1) + dt;
            dt_history(i) = dt;
            
            % Update solution
            u_prev = u_int;
            u_int = u_new;
            time = time + dt;
            
            % Reset rejection counter
            consecutive_rejections = 0;
            
            % Visualization every 100 accepted steps
            if mod(n_it, 100) == 0
                figure(fig1);
                imagesc(x, x, real(u_int)); 
                colormap(gray);
                colorbar;
                title(['time = ' num2str(time_vector(i)) ', dt = ' num2str(dt, '%.2e')]);
                
                figure(fig2);
                semilogy(time_vector(1:i), Energy(1,1:i));
                xlabel('Time');
                ylabel('Energy');
                title('Energy Evolution');
                grid on;
                
                figure(fig3);
                semilogy(1:i, dt_history(1:i));
                xlabel('Step Number');
                ylabel('Time Step Size');
                title('Time Step Evolution');
                grid on;
                
                drawnow;
            end
            
            % Progress output
            if mod(n_it, 1000) == 0
                fprintf('Step %d: t=%.4f, dt=%.2e, Energy=%.6e, ΔE=%.2e, FP_iter=%d\n', ...
                    n_it, time, dt, Energy(1,i), Denergy, k_fp);
            end
            
            n_it = n_it + 1;
            i = i + 1;
        else
            % Step rejected - don't update solution, just decrease dt
            consecutive_rejections = consecutive_rejections + 1;
            
            if consecutive_rejections > max_consecutive_rejections
                fprintf('Too many consecutive rejections. Stopping simulation.\n');
                break;
            end
        end
        
        % Update time step for next iteration
        dt = new_dt;
        
    else
        % Fixed point failed to converge
        dt = max(dt * dt_decrease_factor, dt_min);
        consecutive_rejections = consecutive_rejections + 1;
        
        fprintf('Fixed point failed to converge. Reducing dt to %.2e\n', dt);
        
        if consecutive_rejections > max_consecutive_rejections
            fprintf('Too many consecutive rejections. Stopping simulation.\n');
            break;
        end
    end
    
    % Safety check for minimum time step
    if dt < dt_min
        fprintf('Time step became smaller than minimum. Stopping simulation.\n');
        break;
    end
end

% Final visualization and statistics
figure(fig1);
imagesc(x, x, real(u_int)); 
colormap(gray);
colorbar;
title(['Final Pattern at time = ' num2str(time)]);

figure(fig2);
semilogy(time_vector(1:i-1), Energy(1,1:i-1));
xlabel('Time');
ylabel('Energy');
title('Energy Evolution');
grid on;

figure(fig3);
semilogy(1:i-1, dt_history(1:i-1));
xlabel('Step Number');
ylabel('Time Step Size');
title('Time Step Evolution');
grid on;

% Print final statistics
fprintf('\n=== SIMULATION COMPLETED ===\n');
fprintf('Total steps: %d\n', n_it-1);
fprintf('Final time: %.6f\n', time);
fprintf('Final dt: %.2e\n', dt);
fprintf('Final energy: %.6e\n', Energy(1,i-1));
fprintf('Average dt: %.2e\n', mean(dt_history(1:i-1)));
fprintf('Min dt used: %.2e\n', min(dt_history(1:i-1)));
fprintf('Max dt used: %.2e\n', max(dt_history(1:i-1)));
fprintf('Consecutive rejections: %d\n', consecutive_rejections);


pause(10);