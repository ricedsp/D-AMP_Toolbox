function [x_hat,PSNR] = DAMP(y,iters,width,height,denoiser,M_func,Mt_func,PSNR_func,rgb)
% function [x_hat,PSNR] = DAMP(y,iters,width,height,denoiser,M_func,Mt_func,PSNR_func)
% this function implements D-AMP based on any denoiser present in the
% denoise function
%
% Required Input:
%       y       : the measurements 
%       iters   : the number of iterations
%       width   : width of the sampled signal
%       height  : height of the sampeled signal. height=1 for 1D signals
%       denoiser: string that determines which denosier to use. e.g., 'BM3D'
%       M_func  : function handle that projects onto M. Or a matrix M.
%
% Optional Input:
%       Mt_func  : function handle that projects onto M'.
%       PSNR_func: function handle to evaluate PSNR
%
% Output:
%       x_hat   : the recovered signal.
%       PSNR    : the PSNR trajectory.

if (nargin>=7)&&(~isempty(Mt_func)) % function handles
    M=@(x) M_func(x);
    Mt=@(z) Mt_func(z);
else % explicit Matrix
    M=@(x)M_func*x;
    Mt=@(z)M_func'*z;
end
if (nargin<8)||isempty(PSNR_func) % no PSNR trajectory
    PSNR_func = @(x) nan;
end

if nargin<9
    rgb=false;
end


denoi=@(noisy,sigma_hat) denoise(noisy,sigma_hat,width,height,denoiser,rgb);

if rgb
	n=3*width*height;
else
	n=width*height;
end
m=length(y);

z_t=y;
x_t=zeros(n,1);

PSNR=zeros(1,iters);
for i=1:iters
    pseudo_data=Mt(z_t)+x_t;
    sigma_hat=sqrt(1/m*sum(abs(z_t).^2));
    x_t=denoi(pseudo_data,sigma_hat);
    PSNR(i) = PSNR_func(x_t);
    eta=randn(1,n);
    epsilon=max(pseudo_data)/1000+eps;
    div=eta*((denoi(pseudo_data+epsilon*eta',sigma_hat)-x_t)/epsilon);
    z_t=y-M(x_t)+1/m.*z_t.*div;
end
if rgb
	x_hat=reshape(x_t,[height width,3]);
else
	x_hat=reshape(x_t,[height width]);
end
end
