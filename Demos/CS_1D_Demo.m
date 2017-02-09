%Demonstrates compressively sampling and recovery of 1D piecewise constant
%signal using AMP based on Haar Wavelets and D-AMP based on the NLM
%denoising algorithm.

addpath(genpath('..'));

%Generate Signal
x_0=zeros(256,1);
x_0(1:30)=1;
x_0(100:150)=1;
x_0(180:200)=1;

%Parameters
n=length(x_0(:));
m=round(n/3);
iters=30;

%Generate Gaussian Measurement Matrix
phi=randn(m,n);
for j = 1:n
    phi(:,j) = phi(:,j) ./ sqrt(sum(abs(phi(:,j)).^2));
end

%Generate function handles to be used by AMP to project measurements to and
%from sparsifying wavelet basis
h=daubcqf(2);
M_fp=@(x) phi*psi_fp(x,h,n,0);
Mt_fp=@(z) psit_fp((phi'*z),h,n,0);

%Compressively sample the signal
y=phi*x_0;

%Recover Signal using AMP and D-AMP.  AMP recovers wavelet coefficients
%which are then transformed back into the original domain.
x_hat1_coefs=AMP(y,iters,n,M_fp,Mt_fp);
x_hat1=psi_fp(x_hat1_coefs,h,n,0);
x_hat2=DAMP(y,iters,n,1,'NLM',phi)';

%Plot Results
h4=figure; 
subplot(1,2,1); hold; plot(x_0,'-.r'); plot(x_hat1,'k*'); hold;  l2=legend('True','AMP'); axis([0 300 -.5 1.5])   
subplot(1,2,2);hold; plot(x_0,'-.r'); plot(x_hat2,'k*'); hold; l2=legend('True','NLM-AMP'); axis([0 300 -.5 1.5])   
