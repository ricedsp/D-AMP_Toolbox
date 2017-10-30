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
M1=randn(m,n)+1i*randn(m,n);
for j = 1:n
    M1(:,j) = M1(:,j) ./ sqrt(sum(abs(M1(:,j)).^2));
end

%Generate Coded Diffraction Pattern Handle
signvec = exp(1i*2*pi*rand(n,1));
inds=[1;randsample(n-1,m-1)+1];
I=speye(n);
SubsampM=I(inds,:);
M2=@(x) SubsampM*reshape(fft2(reshape(bsxfun(@times,signvec,x(:)),[height,width])),[n,1])*(1/sqrt(n))*sqrt(n/m);
Mt2=@(x) bsxfun(@times,conj(signvec),reshape(ifft2(reshape(SubsampM'*x(:),[height,width])),[n,1]))*sqrt(n)*sqrt(n/m);
M2_norm2=n;

%Compute magnitudes of compressive samples of the signal
w=sqrt(wvar)*(randn(m,1)+1i*randn(m,1));
y1=abs(M1*x_0(:)+w);
y2=abs(M2(x_0(:))+w);

%Recover Signal using D-prGAMP algorithm
PNSR_func=@(x) PSNR(x_0(:),abs(x(:)));
[x_hat1, PSNR_hist1] = DprGAMP(y1,iters,width,height,denoiser,M1,[],[],Beta_damp,wvar,x_init,PNSR_func);
[x_hat2, PSNR_hist2] = DprGAMP(y2,iters,width,height,denoiser,M2,Mt2,M2_norm2,Beta_damp,wvar,x_init,PNSR_func);

%D-prGAMP Recovery Performance
display([num2str(SamplingRate*100),'% Sampling ', denoiser, '-prGAMP Gaussian Measurements Reconstruction PSNR=',num2str(PSNR_hist1(end))])
display([num2str(SamplingRate*100),'% Sampling ', denoiser, '-prGAMP Coded Diffraction Pattern Reconstruction PSNR=',num2str(PSNR_hist2(end))])

%Plot Recovered Signals
subplot(1,3,1);
imshow(uint8(x_0));title('Original Image');
subplot(1,3,2);
imshow(uint8(abs(x_hat1)));title([denoiser, '-prGAMP Gaussian Measurements']);
subplot(1,3,3);
imshow(uint8(abs(x_hat2)));title([denoiser, '-prGAMP Coded Diffraction Pattern']);

