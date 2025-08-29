% -- CUSTOM SET UP --
gridsize = 1;
N = 300;                   % Number of Gridpoints
dt = 1/10;               % Reduced initial time step
c0 = 9/32;                 % Normalization constant for double well

% CORRECTED PARAMETERS
epsilon = 1/22;
gamma = 1/200;             % Changed from 1/100 to match paper

th = 1.0;                    % Thickness parameter
Nmax = 40;                 % Max iterations for fixed point
tol = 10^(-6);              % Tighter tolerance for fixed point convergence

stop_crit = 10^(-8);       % Tighter stopping criterion
max_it = 50000;

% -- GENERIC SET UP --
x = gridsize/N * (0:N-1);
k = [0:N/2-1 -N/2:-1];     % Proper wave number vector for FFT

% (2*pi/gridsize) *

[xi, eta] = ndgrid(k, k);   % 2D wave numbers
modk2 = xi.^2 + eta.^2;
modk = sqrt(modk2);

% IMPROVED SIGMA FUNCTION
function sig = sigma(A)
    sig = zeros(size(A));
    % Handle zero frequency properly
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
    
    % Constant term for iteration - CORRECTED
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
        
        % Improved numerical stability
        fft_nl = fft2(nl_term);
        u_int2 = real(ifft2(dt * fft_nl ./ denominator) + CT);
        
        error = max(max(abs(u_int2 - u_int)));
        
        % Add relaxation for better convergence
        alpha = 0.7;  % relaxation parameter
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
    ftu = fft2(u) / N^2;  % Proper normalization
    
    S = sigma(th * modk);
    
    % CORRECTED energy calculation
    kinetic_energy = 0.5 * sum(sum((S + gamma * epsilon * modk2) .* abs(ftu).^2));
    potential_energy = (gamma / epsilon) * sum(sum(W)) / N^2;
    
    discrete_energy_value = potential_energy + kinetic_energy;
end

function W = double_well(u, c0)
    u2 = 1 - abs(u).^2;
    W = c0 * (u2.^2);
end

% IMPROVED INITIAL CONDITION
% Use smaller amplitude random initial condition
amplitude = 0.1;
u0 = amplitude * (2*rand(N,N) - 1) + 1i * amplitude * (2*rand(N,N) - 1);

% Add small perturbation to break symmetry
[x1, x2] = ndgrid(x, x);
perturbation = 0.01 * sin(2*pi*x1) .* cos(2*pi*x2);

% u0 = u0 + perturbation;

% CORRECTED LINEAR OPERATOR
% Remove the (2*pi)^2 factor - it should be included in the wave numbers

L = gamma * epsilon * modk2 + sigma(th * modk);

%(2*pi)^2

% Ensure proper handling of zero frequency
L(1,1) = sigma(0);  % This should be 1

% -- TIME STEPPING LOOP INITIALIZATION --
n_it = 1;
time = 0;
i = 1;
u_int = u0;
Du = 1;
Denergy = 1000;

% -- TIME VECTOR VALUES --
time_vector = zeros(max_it, 1);
time_vector(1) = 0;

% -- ENERGY COMPUTATION - INITIALIZATION --
Energy = zeros(1, max_it);
Energy(1,1) = energy_value(gamma, epsilon, N, u0, th, modk, modk2, c0);
i = 2;

fig1 = figure(1);
fig2 = figure(2);

% -- TIME STEPPING LOOP --
while (n_it < max_it) && (abs(Denergy) > stop_crit)
    [k_fp, u, err, conv] = fixpt(u_int, L, dt, N, epsilon, gamma, Nmax, tol, c0);
    
    if conv == 1
        Energy(1,i) = energy_value(gamma, epsilon, N, u, th, modk, modk2, c0);
        Du = max(max(abs(u - u_int)));
        %Denergy = abs(Energy(1,i-1) - Energy(1,i));
        Denergy = Energy(1,i-1) - Energy(1,i);
        disp(Denergy);
        
        time_vector(i) = time_vector(i-1) + dt;
        
        % Visualization every 100 iterations to speed up
        if mod(n_it, 100) == 0
            figure(fig1);
            imagesc(x, x, real(u)); 
            colormap(gray);
            colorbar;
            title(['time = ' num2str(time_vector(i))]);
            
            figure(fig2);
            semilogy(time_vector(1:i), Energy(1,1:i));
            xlabel('Time');
            ylabel('Energy');
            title('Energy Evolution');
            
            drawnow;
        end
        
        % Update values
        u_int = u;
        n_it = n_it + 1;
        i = i + 1;
        
        % Print progress
        if mod(n_it, 1000) == 0
            fprintf('Iteration %d, Time = %.4f, Energy = %.6e, Energy Change = %.6e\n', ...
                n_it, time_vector(i-1), Energy(1,i-1), Denergy);
        end
        
    elseif conv == 0
        dt = dt * 0.5;  % Reduce time step more conservatively
        fprintf('Reducing time step to %.2e\n', dt);
        
        if dt < 1e-15
            warning('Time step became too small. Stopping simulation.');
            break;
        end
    end
end

% Final visualization
figure(fig1);
imagesc(x, x, real(u)); 
colormap(gray);
colorbar;
title(['Final Pattern at time = ' num2str(time_vector(i-1))]);

figure(fig2);
semilogy(time_vector(1:i-1), Energy(1,1:i-1));
xlabel('Time');
ylabel('Energy');
title('Energy Evolution');
grid on;


filename_image = sprintf('image_N=%d_max_it=%d_gamma=%d_eps=%d_dt=%d_th=%d.png',N,max_it,gamma,epsilon,dt,th);
saveas(fig1, filename_image);
filename_energy = sprintf('energy_N=%d_max_it=%d_gamma=%d_eps=%d_dt=%d_th=%d.png',N, max_it,gamma,epsilon,dt,th);
saveas(fig2, filename_energy);