function [x_tplus1,z_tplus1,pseudo_data] = DIT_oneIter(y,z_t,x_t,width,height,denoiser,M_func,Mt_func)
% function x_hat = DIT_oneIter(y,z_t,x_t,width,height,denoiser,M_func,Mt_func)
% This function computes one iteration of the D-IT algorithm
% Input:
%       y       : the measurements
%       z_t     : current residual
%       x_t     : current signal estimate
%       iters   : the number of iterations
%       width   : width of the sampled signal
%       height  : height of the sampeled signal. height=1 for 1D signals
%       denoiser: string that determines which denosier to use. e.g.
%       denoiser='BM3D'
%       M_func  : function handle that projects onto M. Or a matrix M.
%       Mt_func : function handle that projects onto M'. Or no entry
%Output:
%       x_tplus1 : next signal estimate.
%       z_tplus1 : next residual

if nargin==8%function
    M=@(x) M_func(x);
    Mt=@(z) Mt_func(z);
else%Matrix
    M=@(x)M_func*x;
    Mt=@(z)M_func'*z;
end
denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,width,height,denoiser);

n=width*height;
m=length(y);

pseudo_data=Mt(z_t)+x_t;
sigma_hat=sqrt(1/m*sum(abs(z_t).^2));
x_tplus1=denoi(pseudo_data,sigma_hat);
z_tplus1=y-M(x_tplus1);
end