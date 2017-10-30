function [x_hat, PSNR] = DprGAMP(y,iters,width,height,denoiser,A_func,At_func,A_norm2,Beta,wvar,x_init,PSNR_func)
% function [x_hat, PSNR] = DprGAMP(y,iters,width,height,denoiser,A_mat,Beta,wvar,x_init,PSNR_func)
% this function implements D-prGAMP based on any denoiser present in the
% denoise function
% Input:
%       y       : the measurements 
%       iters   : the number of iterations
%       width   : width of the sampled signal
%       height  : height of the sampeled signal. height=1 for 1D signals
%       denoiser: string that determines which denosier to use. e.g.
%       denoiser='BM3D'
%       A_func : Measurement matrix A or forward operator function handle
%       At_func: Backward operator function handle
%       A_norm2: Frobenius norm squared of the meaurement matrix
%       Beta   : damping parameter (small beta is slow but robust)
%       wvar   : an estimate of the measurement noise
%       x_init : initial guess for x
%Output:
%       x_hat   : the recovered signal
%       PSNR    : the PSNR trajectory. 

if (nargin>=7)&&(~isempty(At_func)) % function handles
    A=@(x) A_func(x);
    At=@(z) At_func(z);
    if isempty(A_norm2)
        error('The frobenius norm squared of A is required to use function handles');
    end
else % explicit Matrix
    A=@(x)A_func*x;
    At=@(z)A_func'*z;
    A_norm2=norm(A_func,'fro')^2;
end
if (nargin<12)||isempty(PSNR_func) % no PSNR trajectory
    PSNR_func = @(x) nan;
end

g_out=@(phat,pvar,y,sigma2_w) g_out_phaseless(phat,pvar,y,sigma2_w);
g_in=@(noisy,sigma_hat) denoise(noisy,sigma_hat,width,height,denoiser);

n=width*height;
m=length(y);

x=x_init;
sigma2_x=var(x_init);
s=eps*ones(m,1);
sigma2_p=eps;
x_bar=x_init;
sigma2_s=eps;

PSNR=zeros(1,iters);
for i=1:iters
    sigma2_p=Beta*A_norm2/m*mean(abs(sigma2_x))+(1-Beta)*sigma2_p;
    sigma2_p=max(sigma2_p,eps);
    p=A(x)-sigma2_p*s;
    [g,dg] = g_out(p,sigma2_p,y,wvar);
    s=Beta*g+(1-Beta)*s;
    sigma2_s=-Beta*dg+(1-Beta)*sigma2_s;
    sigma2_r=1/(A_norm2/n*sigma2_s);
    x_bar=Beta*x+(1-Beta)*x_bar;
    r=x_bar+sigma2_r*At(s);
    x=g_in(real(r),sqrt(sigma2_r));
    PSNR(i) = PSNR_func(abs(x));
    eta=randn(1,n);
    epsilon_in=max(abs(r))/1000+eps;
    div_in=eta*((g_in(real(r)+epsilon_in*eta',sqrt(sigma2_r))-x)/epsilon_in);
    sigma2_x=max(sigma2_r*div_in/n,0);
end
x_hat=reshape(x,[height width]);
end
function [g,dg] = g_out_phaseless(phat,pvar,y_abs,sigma2_w)
%Borrowed from GAMP toolbox. http://gampmatlab.wikia.com/wiki/Generalized_Approximate_Message_Passing
        phat_abs=abs(phat);
        B = 2*y_abs.*phat_abs./(sigma2_w+pvar);
        I1overI0 = min( B./(sqrt(B.^2+4)), ...
            B./(0.5+sqrt(B.^2+0.25)) );%upper bounds (11)&(16) from Amos
        y_sca = y_abs./(1+sigma2_w./pvar);
        phat_sca = phat_abs./(1+pvar./sigma2_w);
        zhat = (phat_sca + y_sca.*I1overI0).*sign(phat);
        sigma2_z = y_sca.^2 + phat_sca.^2 ...
            + (1+B.*I1overI0)./(1./sigma2_w+1./pvar) ...
            - abs(zhat).^2;
        g=1/pvar*(zhat-phat);
        dg=1/pvar*(mean(sigma2_z)/pvar-1);%Using uniform variance.
end