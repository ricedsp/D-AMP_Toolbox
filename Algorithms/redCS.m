%   Solve the RED problem
%           min 1/2\|b-Ax\|^2+lambda/2x'*(x-f(x)) where f(x)
%           is a denoiser
%   using the solver FASTA.  
% 
% RED reference: Romano, Yaniv, Michael Elad, and Peyman Milanfar. "The little engine that could: Regularization by denoising (RED)." SIAM Journal on Imaging Sciences 10.4 (2017): 1804-1844.
%
%  Inputs:
%    A   : A matrix or function handle
%    At  : The adjoint/transpose of A
%    b   : A column vector of measurements
%    x0  : Initial guess of solution, often just a vector of zeros
%    opts: Optional inputs to FASTA
%
%   For this code to run, the solver "fasta.m" must be in your path.
%
%   For more details, see the FASTA user guide, or the paper "A field guide
%   to forward-backward splitting with a FASTA implementation."
%
%   Copyright: Tom Goldstein, 2014.
%   Modified by Chris Metzler 2018.
%   Google's RED implementation can be found here: https://github.com/google/RED

function [ solution, outs ] = redCS( A,At,b,x0,opts,prox_ops)


%%  Check whether we have function handles or matrices
if ~isnumeric(A)
    assert(~isnumeric(At),'If A is a function handle, then At must be a handle as well.')
end
%  If we have matrices, create handles just to keep things uniform below
if isnumeric(A)
    At = @(x)A'*x;
    A = @(x) A*x;
end

%  Check for 'opts'  struct
if ~exist('opts','var') % if user didn't pass this arg, then create it
    opts = [];
end


%%  Define ingredients for FASTA
%  Note: fasta solves min f(Ax)+g(x).

%  f(z) = .5 ||z - b||^2
f    = @(z) .5*norm(z-b,'fro')^2;
grad = @(z) z-b;

% denoi = @(noisy,sigma_hat) denoise(noisy,sigma_hat,width,height,denoiser);
denoi = @(noisy) denoise(real(noisy),prox_ops.sigma_hat,prox_ops.width,prox_ops.height,prox_ops.denoiser);

% g = @(x) prox_ops.lambda/2*x'*(x-denoi(x));%Red regularizer
g = @(x) prox_ops.lambda/2*real(x)'*(real(x)-denoi(x));%Red regularizer

% proxg(z,t) = argmin .5||x-z||^2+t*g(x)
prox = @(z,t) iterative_prox_map(z,t,denoi,prox_ops);

%% Call solver
[solution, outs] = fasta(A,At,f,grad,g,prox,x0,opts);

end

function x = iterative_prox_map(z,t,denoi,opts)
%     epsilon=1e-3;
    lambda=opts.lambda;
	x=z;
    prox_iters=opts.prox_iters;
    for iters=1:prox_iters
%         x=(1/(1+t*lambda))*(z+t*lambda/2*denoi(x)+t*lambda/2*(denoi((1+epsilon)*x)-denoi(x))/epsilon);%Monte Carlo approx
        x=(1/(1+t*lambda))*(z+t*lambda*denoi(x));
    end
end