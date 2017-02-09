function x_hat = DIT(y,iters,width,height,denoiser,M_func,Mt_func)
% function x_hat = DIT(y,iters,width,height,denoiser,M_func,Mt_func)
% this function implements D-IT based on any denoiser present in the
% denoise function
% Input:
%       y       : the measurements 
%       iters   : the number of iterations
%       width   : width of the sampled signal
%       height  : height of the sampeled signal. height=1 for 1D signals
%       denoiser: string that determines which denosier to use. e.g.
%       denoiser='BM3D'
%       M_func  : function handle that projects onto M. Or a matrix M.
%       Mt_func : function handle that projects onto M'. Or no entry
%Output:
%       x_hat   : the recovered signal.

if nargin==7%function
    M=@(x) M_func(x);
    Mt=@(z) Mt_func(z);
else%Matrix
    M=@(x)M_func*x;
    Mt=@(z)M_func'*z;
end
denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,width,height,denoiser);

n=width*height;
m=length(y);

z_t=y;
x_t=zeros(n,1);

for i=1:iters
    pseudo_data=Mt(z_t)+x_t;
    sigma_hat=sqrt(1/m*sum(abs(z_t).^2));
    x_t=denoi(pseudo_data,sigma_hat);
    z_t=y-M(x_t);
end
x_hat=reshape(x_t,[height width]);
end
