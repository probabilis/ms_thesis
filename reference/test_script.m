close all;
clear;

if 1 % cat;
  
  A = double(imread('cat_annotated.png'))/255;
  I = double(imread('cat.jpg'))/255;
  
  m = double(A(:,:,1) == 1 & A(:,:,2) == 0 & A(:,:,3) == 0); % red;
  m(:,1) = 1;
  m(:,end) = 1;
  m(1,:) = 1;
  m(end,:) = 1;
  
  f = mean(I,3);
  n = rand(size(I));
  f(m == 0) = n(m == 0);
  
  func = 'l2';
  xin = f;
  imwrite(f, 'cat_input.png');
end


if 0 % disk with concavity
  M = 100;
  N = 100;
  
  f1 =  zeros(M,N);
  f1(M/2,N/2) = 1;
  f1 = double(bwdist(f1) < M/4);
  
  f2 =  zeros(M,N);
  %f2(M/2:end,:) = 1.0;
  
  f2(round(M/1.2),N/2) = 1;
  f2 = double(bwdist(f2) < M/4);
  
  f = max(0,f1-f2);
  m = 1-f2; m(end-10:end,:) = 1;
 
  func = 'l2';
  xin = f;
end


%%
if 0
  f = mean(double(imread('lady.png'))/255,3);
  f = imfilter(f, fspecial('gaussian', [5 5], 1),'symmetric');
  f = imresize(f, 0.5);
  [fx, fy] = gradient(f);
  norm_grad = sqrt(fx.^2  + fy.^2);
  g = ones(size(f));
  %g(norm_grad > 0.1) = 0.2;
  
  %f(:,1) = 0;
  %f(1,:) = 0;
  %f(:,end) = 0;
  %f(end,:) = 0;
  m = f*0; %double(f == 0);
  func = 'l2';
  xin = f;
  
end


%%
if 0
  f = mean(double(imread('star.png'))/255,3);
  f = imfilter(f, fspecial('gaussian', [5 5], 1),0);
  f = imresize(f, 1);
  [fx, fy] = gradient(f);
  norm_grad = sqrt(fx.^2  + fy.^2);
  g = ones(size(f));
  g(norm_grad > 0.1) = 0.2;
  
  f(:,1) = 1;
  f(1,:) = 1;
  f(:,end) = 1;
  f(end,:) = 1;
  m = double(f == 1);
  func = 'l2';
  xin = f;
  
end

%%
if 0
  f = mean(double(imread('triangle.jpg'))/255,3);
  f = imfilter(f, fspecial('gaussian', [5 5], 1),'symmetric');
  f = imresize(f, 1);
  [fx, fy] = gradient(f);
  norm_grad = sqrt(fx.^2  + fy.^2);
  g = ones(size(f));
  g(norm_grad > 0.1) = 0.2;
  
  f(:,1) = 0;
  f(1,:) = 0;
  f(:,end) = 0;
  f(end,:) = 0;
  m = double(f == 0);
  func = 'l2';
  xin = f;
  
end

%%
if 0
  f = mean(double(imread('cube.png'))/255,3);
  f = imfilter(f, fspecial('gaussian', [5 5], 1),'symmetric');
  f = imresize(f, 1);
  [fx, fy] = gradient(f);
  norm_grad = sqrt(fx.^2  + fy.^2);
  g = ones(size(f));
  g(norm_grad > 0.1) = 0.3;
  
  f(:,1) = 0;
  f(1,:) = 0;
  f(:,end) = 0;
  f(end,:) = 0;
  m = double(f == 0);
  func = 'l2';
  xin = f;
  
end


%%
 
lambda = 1;
epsilon = 10;
alpha = 0.01;
beta = 1;
delta = 1;
maxiter = 30000;

[x] = elastica_ipiano(xin, f, m, ones(size(f)), func, epsilon, lambda, alpha, beta, delta, maxiter);