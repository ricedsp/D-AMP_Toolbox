%Demonstrates compressively sampling and D-AMP recovery of an image.

addpath(genpath('..'));

%Parameters
filename='peppers_color.tiff';
SamplingRate=.2;
iters=30;
imsize=128;

rgb=true;
if rgb
    denoiser1='CBM3D';
    ImIn=double(imread(filename));
else
    denoiser1='BM3D';%Available options are NLM, Gauss, Bilateral, BLS-GSM, BM3D, fast-BM3D, and BM3D-SAPCA 
    ImIn=double(rgb2gray(imread(filename)));
end
x_0=imresize(ImIn,imsize/size(ImIn,1));
[height, width, ~]=size(x_0);
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
x_hat1 = DAMP(y,iters,height,width,denoiser1,M,[],[],rgb);

%D-AMP Recovery Performance
performance1=PSNR(x_0(:),x_hat1(:));
[num2str(SamplingRate*100),'% Sampling ', denoiser1, '-AMP Reconstruction PSNR=',num2str(performance1)]

%Plot Recovered Signals
subplot(1,2,1);
imshow(uint8(x_0));title('Original Image');
subplot(1,2,2);
imshow(uint8(x_hat1));title([denoiser1, '-AMP']);
