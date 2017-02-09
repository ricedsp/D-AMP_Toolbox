%Generates state evolution and compares to observed MSEs for D-AMP and D-IT algorithms

addpath(genpath('..'));

%Parameters
denoiser='BM3D';
SamplingRate=.2;
noise_sig=0;
imsize=128;
filename='barbara.png';
iters=30;
N=5;%Number of tests to run.  Must be at least 2

Im=double(imread(filename));
x_0=imresize(Im,imsize/size(Im,1));
[height, width]=size(x_0);
n=length(x_0(:));
m=round(n*SamplingRate);

%Generate State Evolution and compute intermediate MSEs of D-AMP and D-IT
True_DAMP_MSE_array=zeros(N,iters);
True_DIT_MSE_array=zeros(N,iters);
Predicted_MSE_array=zeros(N,iters);
True_DAMP_MSE_array(:,1)=mean(x_0(:).^2);
True_DIT_MSE_array(:,1)=mean(x_0(:).^2);
Predicted_MSE_array(:,1)=mean(x_0(:).^2);
for i=1:N
    i
    M=randn(m,n);
    for j = 1:n
        M(:,j) = M(:,j) ./ sqrt(sum(abs(M(:,j)).^2));
    end
    y=M*x_0(:)+noise_sig*randn(m,1);
    x_t=zeros(n,1);x_t2=zeros(n,1);
    z_t=y;
    z_t2=y;
    for iter=2:iters
            [x_tplus1,z_tplus1,NA] = DAMP_oneIter(y,z_t,x_t,width,height,denoiser,M);
            z_t=z_tplus1; x_t=x_tplus1;
            [x_tplus12,z_tplus12,NA] = DIT_oneIter(y,z_t2,x_t2,width,height,denoiser,M);
            z_t2=z_tplus12; x_t2=x_tplus12;
            True_DAMP_MSE_array(i,iter)=mean((x_0(:)-x_tplus1(:)).^2);
            True_DIT_MSE_array(i,iter)=mean((x_0(:)-x_tplus12(:)).^2);
            Predicted_MSE_array(i,iter)= DAMP_SE_Prediction(x_0(:), Predicted_MSE_array(i,iter-1), m,n,noise_sig,denoiser,width,height);
    end
end
True_DAMP_MSE=mean(True_DAMP_MSE_array)';
True_DAMP_MSE_std=std(True_DAMP_MSE_array)';
True_DIT_MSE=mean(True_DIT_MSE_array)';
True_DIT_MSE_std=std(True_DIT_MSE_array)';
Predicted_MSE=mean(Predicted_MSE_array)';
Predicted_MSE_std=std(Predicted_MSE_array)';

%Plot Results
h=figure; 
hold
errorbar(0:29,Predicted_MSE,Predicted_MSE_std,'-.b');
errorbar(0:29,True_DAMP_MSE,True_DAMP_MSE_std,'--g');
errorbar(0:29,True_DIT_MSE,True_DIT_MSE_std,'-r');
title([denoiser,'-AMP and ', denoiser '-IT State Evolution']);
legend(['Predicted ',num2str(100*SamplingRate),'%'], [denoiser,'-AMP ', num2str(100*SamplingRate),'%'], [denoiser,'-IT ', num2str(100*SamplingRate),'%']);
xlabel('Iteration');
ylabel('MSE');

