%Generates QQplot for D-AMP and D-IT algorithms

addpath(genpath('..'));

%Parameters
denoiser='BM3D';
SamplingRate=1/2;
noise_sig=0;
imsize=128;
filename='barbara.png';

Im=double(imread(filename));
x_0=imresize(Im,imsize/size(Im,1));
% x_0=zeros(256,1);
% x_0(1:30)=1;
% x_0(100:150)=1;
% x_0(180:200)=1;
[height, width]=size(x_0);
iters=30;
n=length(x_0(:));
m=round(n*SamplingRate);

%Generate Measurement Matrix
% randn('seed',5);
M=randn(m,n);
for j = 1:n
    M(:,j) = M(:,j) ./ sqrt(sum(abs(M(:,j)).^2));
end

%Compressively sample signal
y=M*x_0(:)+noise_sig*randn(m,1);

%Reconstruct x and determine D-iT and D-AMP effective noise
x_t1=zeros(n,1);x_t2=zeros(n,1);
z_t1=y;
z_t2=y;
DAMP_effective_error=zeros(n,iters);
DIT_effective_error=zeros(n,iters);
DAMP_effective_error(:,1)=-x_0(:);
DIT_effective_error(:,1)=-x_0(:);
for iter=2:iters
    [x_t1,z_t1,pseudo_dat1] = DAMP_oneIter(y,z_t1,x_t1,width,height,denoiser,M);
    [x_t2,z_t2,pseudo_dat2] = DIT_oneIter(y,z_t2,x_t2,width,height,denoiser,M);
    DAMP_effective_error(:,iter)=pseudo_dat1-x_0(:);
    DIT_effective_error(:,iter)=pseudo_dat2-x_0(:);
end

%Plot Reconstructions
close all;
% h2=figure;subplot(1,2,1);plot(x_t2);title([denoiser,'-IT']);subplot(1,2,2);plot(x_t1);title([denoiser,'-AMP']);
h1=figure;subplot(1,2,1);imshow(uint8(reshape(x_t2,[imsize imsize])));title([denoiser,'-IT']);subplot(1,2,2);imshow(uint8(reshape(x_t1,[imsize imsize])));title([denoiser,'-AMP']);

%Plot QQplots
h2=figure;
k=3;
axislimits=[-5 5 -5 5];
subplot(4,2,2);qqplot([DAMP_effective_error(:,k)/std(DAMP_effective_error(:,k))]);axis(axislimits);title([num2str(denoiser),'-AMP Effective Error Iteration ',num2str(k)]);
subplot(4,2,1);qqplot([DIT_effective_error(:,k)/std(DIT_effective_error(:,k))]);axis(axislimits);title([num2str(denoiser),'-IT Effective Error Iteration ',num2str(k)]);
k=5;
subplot(4,2,4);qqplot([DAMP_effective_error(:,k)/std(DAMP_effective_error(:,k))]);axis(axislimits);title([num2str(denoiser),'-AMP Effective Error Iteration ',num2str(k)]);
subplot(4,2,3);qqplot([DIT_effective_error(:,k)/std(DIT_effective_error(:,k))]);axis(axislimits);title([num2str(denoiser),'-IT Effective Error Iteration ',num2str(k)]);
k=10;
subplot(4,2,6);qqplot([DAMP_effective_error(:,k)/std(DAMP_effective_error(:,k))]);axis(axislimits);title([num2str(denoiser),'-AMP Effective Error Iteration ',num2str(k)]);
subplot(4,2,5);qqplot([DIT_effective_error(:,k)/std(DIT_effective_error(:,k))]);axis(axislimits);title([num2str(denoiser),'-IT Effective Error Iteration ',num2str(k)]);
k=20;
subplot(4,2,8);qqplot([DAMP_effective_error(:,k)/std(DAMP_effective_error(:,k))]);axis(axislimits);title([num2str(denoiser),'-AMP Effective Error Iteration ',num2str(k)]);
subplot(4,2,7);qqplot([DIT_effective_error(:,k)/std(DIT_effective_error(:,k))]);axis(axislimits);title([num2str(denoiser),'-IT Effective Error Iteration ',num2str(k)]);

