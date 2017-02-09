%Demonstrates compressively sampling and D-AMP recovery of an image.

addpath(genpath('..'));

%Parameters
denoiser1='BM3D';%Available options are NLM, Gauss, Bilateral, BLS-GSM, BM3D, fast-BM3D, and BM3D-SAPCA 
denoiser2='fast-BM3D';
filename='barbara.png';
SamplingRate=.2;
iters=30;
imsize=128;

ImIn=double(imread(filename));
x_0=imresize(ImIn,imsize/size(ImIn,1));
[height, width]=size(x_0);
n=length(x_0(:));
m=round(n*SamplingRate);


%Generate Gaussian Measurement Matrix
M=randn(m,n);
for j = 1:n
    M(:,j) = M(:,j) ./ sqrt(sum(abs(M(:,j)).^2));
end

%Compressively sample the image
y=M*x_0(:);

%Recover Signal using D-AMP algorithms
x_hat1 = DAMP(y,iters,height,width,denoiser1,M);
x_hat2 = DAMP(y,iters,height,width,denoiser2,M);

%D-AMP Recovery Performance
performance1=PSNR(x_0,x_hat1);
performance2=PSNR(x_0,x_hat2);
[num2str(SamplingRate*100),'% Sampling ', denoiser1, '-AMP Reconstruction PSNR=',num2str(performance1)]
[num2str(SamplingRate*100),'% Sampling ', denoiser2, '-AMP Reconstruction PSNR=',num2str(performance2)]

%Plot Recovered Signals
subplot(1,3,1);
imshow(uint8(x_0));title('Original Image');
subplot(1,3,2);
imshow(uint8(x_hat1));title([denoiser1, '-AMP']);
subplot(1,3,3);
imshow(uint8(x_hat2));title([denoiser2, '-AMP']);
