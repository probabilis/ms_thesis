% -- CUSTOM SET UP --
gridsize = 1;
N = 128; %512                   % Number of Gridpoints
dt = 1/3000;               % Time step
c0 = 9/32;                 % Normalization constant for double well


epsilon = 1/20;
gamma = 1/400;

th = 1;                    % Thickness parameter
Nmax = 40;                 % Max iterations for fixed point
tol = 10^(-8);                % Tolerance for fixed point convergence
stop_crit = 10^(-8);          % Stopping criterion for time iteration
max_it = 50000;


% -- GENERIC SET UP --
x = gridsize/N * (0:N-1);
k = [0:N/2-1 -N/2:-1];     % Proper wave number vector for FFT
[xi, eta] = ndgrid(k, k);   % 2D wave numbers
modk2 = xi.^2 + eta.^2;
modk = sqrt(modk2);



function [k, u_n, err, conv] = fixpt(u0, L, dt, N, epsilon, gamma, Nmax, tol, c0)
    % Nonlinear function (replaced inline with handle)
    NL = @(u, v, epsilon, gamma, c0) 2 * gamma * c0 / epsilon .* (u + v) .* (1 - (abs(u).^2 + abs(v).^2) / 2);

    % Constant term for iteration
    CT = real(ifft2( (ones(N) - dt/2 * L) ./ (ones(N) + dt/2 * L) .* fft2(u0) ) );

    % Initialization
    j = 1;
    error = 10;
    u_int = u0;
    u_int2 = N * ones(N,N);
    convergence = 0;

    
    disp("L, CT")
    disp(max(max(L)));
    disp(max(max(abs(CT))));
    disp("max u0 and u02")
    disp(mean(mean(abs(u0))));
    disp(mean(mean(abs(u0).^2)));

    % Fixed Point Iteration Loop
    while (j < Nmax) && (error > tol)
        test = NL(u_int, u0, epsilon, gamma, c0);

        disp("NL");
        disp(max(max(abs(test))));

        u_int2 = real(ifft2(fft2(dt * test) ./ (ones(N) + dt/2 * L) ) + CT);
        error = max(max(abs(u_int2 - u_int)));
        disp("error");
        disp(error);
        u_int = u_int2;
        j = j + 1;
    end

    if error < tol
        convergence = 1;
    end

    k = j;
    err = error;
    u_n = u_int2;
    conv = convergence;
end


function [discrete_energy_value] = energy_value(gamma, epsilon, N, u, th, modk, modk2, c0)
    W = double_well(u, c0);
    ftu = 1 / N^2 * fft2(u);

    S = sigma(th * modk);
    disp("max S");
    disp(max(S(:)))
    disp(min(S(:)))
    disp(mean(S(:)))

    discrete_energy_value = (gamma / epsilon) * 1 / N^2 * (sum(sum(W))) + 1/2 * sum(sum((sigma(th * modk) + gamma * epsilon * (modk2)) .* abs(ftu).^2));
end


function W = double_well(u, c0)
    u2 = 1 - abs(u).^2;
    W = c0 * (u2.^2);
end


function sig = sigma(A)
    sig = zeros(size(A));
    small = abs(A) < 1e-12;
    large = ~small;
    sig(small) = 1 - pi * abs(A(small));
    sig(large) = 1 - (1 - exp(-2 * pi * abs(A(large)))) ./ (2 * pi * abs(A(large)));
end


function sig = sigma2(A)
    if (abs(A)< 1.0e-12)
        B = 1-(pi*abs(A));
    else
        B = (1 - exp(-2 * pi * abs(A) ) ) / (2*pi*abs(A));
    end
sig=B;
end





%u0 = (2*rand(N, N) - 1) + (2i * rand(N, N) - 1i);  % Complex initial condition
%u0 = tanh(10 * (rand(N, N) - 0.5)) + 1i * tanh(10 * (rand(N, N) - 0.5));
u0 = randn(N,N) + i * randn(N,N);


disp("max INV FFT");
tmp = fft2(u0);
inv_fft = ifft2(tmp);
disp(max(abs(inv_fft(:))))  % ~O(1)



disp("max NL");
NL = 2 * gamma * c0 / epsilon * (u0 + u0) .* (1 - (abs(u0).^2 + abs(u0).^2) / 2);
disp(max(abs(NL(:))));



L = (2 * pi)^2 * gamma * epsilon * (modk2) + sigma(th * modk);   % Linear operator
Mult = ones(N) - dt/2 * L;           % Fourier space multiplication operator
Inv = ones(N) + dt/2 * L;            % Inverse operator in Fourier space


% -- TIME STEPPING LOOP INITIALIZATION --
n_it = 1;
time = 0;
i = 1;
time = time + dt;
u_int = u0;
Du = 1;
Denergy = 1000;

% -- TIME VECTOR VALUES --
time_vector = zeros(max_it, 1);
time_vector(1) = 0;

% -- ENERGY COMPUTATION - INITIALIZATION --
%Energy = zeros(1, max_it);
[Energy(1,i)] = energy_value(gamma, epsilon, N, u0, th, modk, modk2, c0);
i = i + 1;


% -- TIME STEPPING LOOP --
while abs(Denergy) > stop_crit
    [k_fp, u, err, conv] = fixpt(u_int, L, dt, N, epsilon, gamma, Nmax, tol, c0);
    disp("CONV");
    disp(conv);
    disp("dt");
    disp(dt);
    if conv == 1
        [Energy(1,i)] = energy_value(gamma, epsilon, N, u, th, modk, modk2, c0);
        Du = max(max(abs(u - u_int)));
        Denergy = Energy(1,i-1) - Energy(1,i);
        disp("Diff energy:");
        disp(Denergy);

        time_vector(i) = time_vector(i-1) + dt;

        % -- Pattern visualization --
        fig = imagesc(x, x, u); colormap(gray);
        title(['time = ' num2str(time)]);
        drawnow;
        Frames(i-1) = getframe(gcf); % store frame

        % -- Update values --
        u_int = u;
        time = time + dt;
        n_it = n_it + 1;
        i = i + 1;
        disp(time)
        saveas(fig, 'test_mat.png')
    elseif (conv == 0)
        dt = dt/4;  % reduce time step
        %if dt < 1e-12
           % savefig("test_mat.fig")
        %    saveas('test_mat.png')
        %    error('Time step too small. Exiting.');
        
    end
end








