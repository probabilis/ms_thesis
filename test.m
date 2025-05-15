
A = [1e-13,1e-5; 1e-2,1e-11];

% A = [1,1e-13; 10,1e-13];

% A = randn(2);

function sig = sigma(A)
    if (abs(A)< 1.0e-12)
        B = 1-(pi*abs(A));
        disp("S");
    else
        B = (1 - exp(-2 * pi * abs(A) ) ) / (2*pi*abs(A));
        disp("B");
    end
sig=B;
end

function sig = sigma2(A)
    sig = zeros(size(A));
    small = abs(A) < 1e-12;
    large = ~small;
    sig(small) = 1 - pi * abs(A(small));
    sig(large) = 1 - (1 - exp(-2 * pi * abs(A(large)))) ./ (2 * pi * abs(A(large)));
end

disp(A);
disp("----");
B = sigma(A);
disp(B);
disp("----");
C = sigma2(A);
disp(C);


%C = linspace(1,10,10);
%disp(C);
%FT = fft2(C);
%disp(FT);