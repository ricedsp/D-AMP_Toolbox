function [x_hat,PSNR,estFin,estHist] = VAMP(y,iters,n,M_func,Mt_func,PSNR_func,U_func,Ut_func,d,UtM_func,MtU_func)
% function [x_hat,PSNR] = VAMP(y,iters,n,M_func,Mt_func,PSNR_func)
% This function implements VAMP for sparse recovery
% Input:
%       y       : the measurements 
%       iters   : the number of iterations
%       n       : signal size
%       M_func  : function handle that projects onto A. Or a matrix A.
%       Mt_func : function handle that projects onto A'. Or no entry
%       PSNR_func : optional function handle to evaluate PSNR
% Output:
%       x_hat   : the recovered signal.
%       PSNR    : the PSNR trajectory.

vampOpt = VampSlmOpt;
vampOpt.nitMax = iters;

if (nargin>4)&&(~isempty(Mt_func)) % function handles
    vampOpt.Ah = Mt_func;
    vampOpt.N = n;
end
if (nargin>5)&&(~isempty(PSNR_func)) % error trajectory
    vampOpt.fxnErr = @(x2) PSNR_func(x2);
end
if (nargin>6) % external eigendecomposition [U,D]=eig(M_func(Mt_func(I)))
    vampOpt.U = @(x) U_func(x); % function handle to U
    vampOpt.Uh = @(x) Ut_func(x); % function handle to U'
    vampOpt.d = d; % eigenvalues d=diag(D)
end
if (nargin>9) % external SVD [U,S,V]=svd(M_func) with U matching eigenvectors above
    vampOpt.UhA = @(x) UtM_func(x); % function handle to U'*M = S*V'
    vampOpt.AhU = @(x) MtU_func(x); % function handle to M'*U = V*S'
end

% prepare for VAMP
alf = 1.0;
debias = false; % automatically debias the soft thresholder?
denoi = SoftThreshDMMEstimIn(alf,'debias',debias);

% run VAMP
if nargout==4
  [x_hat,estFin,estHist] = VampSlmEst(denoi,y,M_func,vampOpt);
else
  [x_hat,estFin] = VampSlmEst(denoi,y,M_func,vampOpt);
end
PSNR = estFin.err;
