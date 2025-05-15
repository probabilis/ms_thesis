function [u] = elastica_ipiano(init, f, m, g, func, epsilon, lambda, alpha, beta, delta, maxiter)
  
[M, N] = size(f);
f = f(:);
init = init(:);
m = m(:);

%%
nabla = make_nabla(M,N);
lap = -nabla'*nabla;

u = init;
u_old = u;

L = 1;
for k = 0:maxiter
  
  theta = (k-1)/(k+2);
  
  u_ = u + theta*(u-u_old);
  u_old = u;
  
  %% compute gradient
  W0 = u_.^2.*(1-u_).^2;
  W1 = 2*u_.^3 - 3*u_.^2 + u_;
  W2 = 6*u_.^2-6*u_+1;
  
  curv = W1/epsilon-epsilon*lap*u_;
  curv_ = spdiags(W2,0,M*N,M*N)/epsilon^2 - lap;
  
  grad_tv = curv;  
  grad_curv = curv_*curv;  
  grad = alpha*grad_tv+beta*grad_curv;
 
  Q0 = alpha*(epsilon/2*norm(nabla*u_)^2+1/epsilon/2*sum(W0)) + beta/2/epsilon*norm(curv)^2;
  
  for bt_iter=1:10
    
    tau = 1/L;
    
    %% gradient step
    u = u_ - tau*grad;
  
    %% make backward step
    if strcmp(func, 'l2')
      u = (u + tau*lambda*f.*m)./(1+tau*lambda*m);
    end
    if strcmp(func, 'lin')
      u = max(0, min (1.0, u - tau*lambda*f.*m));
    end
  
    % Compute Energy of quadratic approximation
    Q = Q0 + sum((u-u_).*grad) + L/2*norm(u-u_)^2;
    
    % Compute new energy
    W0 = u.^2.*(1-u).^2;
    W1 = 2*u.^3 - 3*u.^2 + u;
    curv = W1/epsilon-epsilon*lap*u;
  
    E = alpha*(epsilon/2*norm(nabla*u)^2+1/epsilon/2*sum(W0)) + beta/2/epsilon*norm(curv)^2;
    
    if E < Q
      L = L/1.5;
      break;
    else
      L = L*2;
    end
    
  end
  
  if mod(k, 100) == 0
    fprintf('iter=%04d, epsilon=%f, theta=%f, L=%f, E = %f, \n', k, epsilon, theta, L, E);
    poster = [reshape(f, M, N), reshape(u, M, N), norm01(reshape(curv,M,N))];
    sfigure(1); imshow(imresize(poster, 1, 'nearest'),[0 1]); drawnow;
    drawnow;
    xx = reshape(u, M, N);
    imwrite(xx,  sprintf('out/%04d.png', k/100));
  end
  
  % continuation in epsilon
  if mod(k,10) == 0;
    epsilon = max(1, epsilon*0.999);
  end
  
end 