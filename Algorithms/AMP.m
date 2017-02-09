function [x_hat,PSNR] = AMP(y,iters,n,M_func,Mt_func,PSNR_func)
% function [x_hat,PSNR] = AMP(y,iters,n,M_func,Mt_func,PSNR_func)
% This function implements AMP for sparse recovery
% Input:
%       y       : the measurements 
%       iters   : the number of iterations
%       n       : signal size
%       M_func  : function handle that projects onto M. Or a matrix M.
%       Mt_func : function handle that projects onto M'. Or no entry
%       PSNR_func: optional function handle to evaluate PSNR
% Output:
%       x_hat   : the recovered signal.
%       PSNR    : the PSNR trajectory.

if (nargin>=5)&&(~isempty(Mt_func)) % function handles
    M=@(x) M_func(x);
    Mt=@(z) Mt_func(z);
else % explicit Matrix
    M=@(x)M_func*x;
    Mt=@(z)M_func'*z;
end
if (nargin<6)||isempty(PSNR_func) % no PSNR trajectory
    PSNR_func = @(x) nan;
end 


m=length(y);
load('OptimumLambdaSigned.mat');  % this file has the optimal values of lambda
delta_check=delta_vec;
delta=m/n;
lambda=interp1(delta_check,lambda_opt,delta);

z_t=y;
x_t=zeros(n,1);
PSNR=zeros(1,iters);
for iter=1:iters
    pseudo_data=Mt(z_t)+x_t;
    sigma_hat=sqrt(1/m*sum(abs(z_t).^2));
    x_t=(abs(pseudo_data)> lambda*sigma_hat).*(abs(pseudo_data)-lambda*sigma_hat).*sign(pseudo_data);
    PSNR(iter)=PSNR_func(x_t);
    z_t=y-M(x_t)+1/m.*z_t.*length(find(abs(x_t)>0));       
end
x_hat=x_t;
end
