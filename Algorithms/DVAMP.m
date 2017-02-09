function [x_hat,PSNR,estFin,estHist] = DVAMP(y,iters,width,height,denoiser,M_func,Mt_func,PSNR_func,U_func,Ut_func, d, UtM_func,MtU_func)
% function [x_hat,PSNR] = DVAMP(y,iters,width,height,denoiser,M_func,Mt_func,PSNR_func,U_func,Ut_func)
% this function implements D-VAMP based on any denoiser present in the
% denoise function
% Required Input:
%       y       : the measurements 
%       iters   : the number of iterations
%       width   : width of the sampled signal
%       height  : height of the sampled signal. height=1 for 1D signals
%       denoiser: string that determines which denosier to use. e.g., 'BM3D'
%       M_func  : function handle that projects onto M. Or a matrix M.
%
% Option Input:
%       Mt_func : function handle that projects onto M'. Or a matrix M'.
%       PSNR_func: function handle to evaluate PSNR
%       U_func  : function handle to U in [U,D]=eig(M*M')
%       Ut_func  : function handle to U' in [U,D]=eig(M*M')
%       d : vector of diagonal elements from D in [U,D]=eig(M*M')
% 
% Output:
%       x_hat   : the recovered signal.
%       PSNR    : the PSNR trajectory.

n=width*height;
m=length(y);

vampOpt = VampSlmOpt;
vampOpt.nitMax = iters;
vampOpt.learnNoisePrec = true;
vampOpt.divChange = 0.1; % relative distance to probe when computing divergence
%vampOpt.tol = 1e-2; % early stopping

if (nargin>6)&&(~isempty(Mt_func)) % function handles
    vampOpt.Ah = Mt_func;
    vampOpt.N = n;
end
if (nargin>7)&&(~isempty(PSNR_func)) % error trajectory
    vampOpt.fxnErr = @(x2) PSNR_func(x2);
end
if nargin>8 % external eigendecomposition [U,D]=eig(M*M')
    vampOpt.U = @(x) U_func(x); % function handle to U
    vampOpt.Uh = @(x) Ut_func(x); % function handle to U'
    vampOpt.d = d; % eigenvalues d=diag(D)
end
if nargin>11 % fast fxn handles to U'M and M'*U
    vampOpt.UhA = @(x) UtM_func(x); % function handle to U'*M = S*V'
    vampOpt.AhU = @(x) MtU_func(x); % function handle to M'*U = V*S'
end

denoi = @(noisy,sigma2_hat) denoise(noisy,sqrt(sigma2_hat(1,:)),width,height,denoiser);

% run VAMP
if nargout==4
  [x_t,estFin,estHist] = VampSlmEst(denoi,y,M_func,vampOpt);
else
  [x_t,estFin] = VampSlmEst(denoi,y,M_func,vampOpt);
end
x_hat=reshape(x_t,[height width]);
PSNR = estFin.err;
