%Demonstrates compressive phase retrieval D-prGAMP recovery of an image.

addpath(genpath('..'));

%Parameters
denoiser='BM3D';%Available options are NLM, Gauss, Bilateral, BLS-GSM, BM3D, fast-BM3D, and BM3D-SAPCA 
filename='barbara.png';
SamplingRate=.6;
iters=100;
imsize=128;
Beta_damp=.1;
wvar=.1;

ImIn=double(imread(filename));
x_0=imresize(ImIn,imsize/size(ImIn,1));
[height, width]=size(x_0);
n=length(x_0(:));
m=round(n*SamplingRate);
x_init=255*rand(n,1);

%Generate Gaussian Measurement Matrix
M=randn(m,n)+1i*randn(m,n);
for j = 1:n
    M(:,j) = M(:,j) ./ sqrt(sum(abs(M(:,j)).^2));
end

%Compute magnitudes of compressive samples of the signal
w=sqrt(wvar)*(randn(m,1)+1i*randn(m,1));
y=abs(M*x_0(:)+w);

%Recover Signal using D-prGAMP algorithm
x_hat = DprGAMP(y,iters,width,height,denoiser,M,Beta_damp,wvar,x_init);

%D-prGAMP Recovery Performance
performance=PSNR(x_0,abs(x_hat));
[num2str(SamplingRate*100),'% Sampling ', denoiser, '-prGAMP Reconstruction PSNR=',num2str(performance)]

%Plot Recovered Signals
subplot(1,2,1);
imshow(uint8(x_0));title('Original Image');
subplot(1,2,2);
imshow(uint8(abs(x_hat)));title([denoiser, '-prGAMP']);
