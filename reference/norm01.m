function [x] = norm01(x)

x = x-min(x(:));
x = x/max(x(:));