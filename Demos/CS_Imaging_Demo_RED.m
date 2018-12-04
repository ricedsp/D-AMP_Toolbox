%Demonstrates compressively sampling and regularization by denoising (RED) recovery of an image.
%Requires  matconvnet and fasta in the path
%Google's RED implementation can be found here: https://github.com/google/RED
% RED reference: Romano, Yaniv, Michael Elad, and Peyman Milanfar. "The little engine that could: Regularization by denoising (RED)." SIAM Journal on Imaging Sciences 10.4 (2017): 1804-1844.

addpath(genpath('..'))
addpath('~/matconvnet/matlab');
addpath(genpath('~/fasta-matlab'));
% addpath(genpath('~/gampmatlab'));

denoiser1='DnCNN';%Available options are NLM, Gauss, Bilateral, BLS-GSM, BM3D, fast-BM3D, BM3D-SAPCA, and DnCNN
denoiser2='DnCNN';
filename='boat.png';
SamplingRate=.1;
RED_iters=200;
AMP_iters=10;
% VAMP_iters=10;
imsize=128; 
n_DnCNN_layers=20;%Other option is 17
LoadNetworkWeights(n_DnCNN_layers);

ImIn=double(imread(filename));
x_0=imresize(ImIn,imsize/size(ImIn,1));
[height, width]=size(x_0);
n=length(x_0(:));
m=round(n*SamplingRate);
errfxn = @(x_hat) PSNR(x_0,reshape(x_hat,[height width]));


% %Generate Gaussian Measurement Matrix
% M_matrix=1/sqrt(m)*randn(m,n);
% M=@(x) M_matrix*x(:);
% Mt=@(x) M_matrix'*x(:);
% U=[];
% Ut=[];
% d=[];


% %Generate Coded Diffraction Pattern Measurement Matrix
% signvec = exp(1i*2*pi*rand(n,1));
% inds=[1;randsample(n-1,m-1)+1];
% I=speye(n);
% SubsampM=I(inds,:);
% M=@(x) SubsampM*reshape(fft2(reshape(bsxfun(@times,signvec,x(:)),[height,width])),[n,1])*(1/sqrt(n))*sqrt(n/m);
% Mt=@(x) bsxfun(@times,conj(signvec),reshape(ifft2(reshape(SubsampM'*x(:),[height,width])),[n,1]))*sqrt(n)*sqrt(n/m);
% U=@(x) x(:);
% Ut= @(x) x(:);
% d=ones(m,1)*n/m;

% %Generate Real-valued Measurement Matrix with Fast Transformation and
% %approximately i.i.d. Gaussian distribution
% signvec = 2*round(rand(n,1))-1;
% inds=[1;randsample(n-1,m-1)+1];
% I=speye(n);
% SubsampM=I(inds,:);
% M=@(x) SubsampM*reshape(dct2(reshape(bsxfun(@times,signvec,x(:)),[height,width])),[n,1])*sqrt(n/m);
% Mt=@(x) bsxfun(@times,conj(signvec),reshape(idct2(reshape(SubsampM'*x(:),[height,width])),[n,1]))*sqrt(n/m);
% U=@(x) x(:);
% Ut= @(x) x(:);
% d=ones(m,1)*n/m;

%Generate (something close to) a Fast JL Transform matrix
signvec = 2*round(rand(n,1))-1;
inds=[1;randsample(n-1,m-1)+1];
I=speye(n);
SubsampM=I(inds,:);
M=@(x) SubsampM*reshape(dct(bsxfun(@times,signvec,x(:))),[n,1])*sqrt(n/m);
Mt=@(x) bsxfun(@times,conj(signvec),reshape(idct(SubsampM'*x(:)),[n,1]))*sqrt(n/m);
U=@(x) x(:);
Ut= @(x) x(:);
d=ones(m,1)*n/m;

% %Generate (something close to) a Fast JL Transform matrix
% signvec = 2*round(rand(n,1))-1;
% inds=[1;randsample(n-1,m-1)+1];
% I=speye(n);
% SubsampM=I(inds,:);
% M=@(x) SubsampM*reshape(fwht(bsxfun(@times,signvec,x(:))),[n,1])*n/sqrt(m);
% Mt=@(x) bsxfun(@times,conj(signvec),reshape(ifwht(SubsampM'*x(:)),[n,1]))*sqrt(1/m);
% U=@(x) x(:);
% Ut= @(x) x(:);
% d=ones(m,1)*n/m;

%Compressively sample the image
y=M(x_0(:));

%Recover signal using RED
%Set RED options
prox_opts=[];
prox_opts.width=width;
prox_opts.height=height;
prox_opts.denoiser=denoiser1;
prox_opts.prox_iters=1;
fasta_opts=[];
fasta_opts.maxIters=RED_iters/2;
fasta_opts.tol=1e-7;
fasta_opts.recordObjective=true;
% fasta_ops.function=errfxn;

x_init=100*rand(size(x_0(:)));

%Recover Signal using RED
prox_opts.lambda=1;
t0=tic;
x_hat1=x_init;
prox_opts.sigma_hat=50;
[x_hat1,out1]  = redCS( M,Mt,y,x_hat1(:),fasta_opts,prox_opts);
prox_opts.sigma_hat=15;
[x_hat1,out2]  = redCS( M,Mt,y,x_hat1(:),fasta_opts,prox_opts);
x_hat1=reshape(x_hat1,[height, width]);
t1=toc(t0);

% figure(4);
% subplot(1,2,1);plot(out1.objective);
% subplot(1,2,2);plot(out2.objective);

%Recover Signal using D-(V)AMP algorithms
t0=tic;[x_hat2,psnr2]  = DAMP(y,AMP_iters,height,width,denoiser1,M,Mt,errfxn);t2=toc(t0);
% t0=tic;[x_hat2,psnr2] = DVAMP(y,VAMP_iters,height,width,denoiser2,M,Mt,errfxn, U, Ut, d);t2=toc(t0);

%D(V)AMP Recovery Performance
performance1=PSNR(x_0,x_hat1);
performance2=PSNR(x_0,x_hat2);
display([num2str(SamplingRate*100),'% Sampling ', denoiser1, '-RED: PSNR=',num2str(performance1),', time=',num2str(t1)])
display([num2str(SamplingRate*100),'% Sampling ', denoiser2, '-AMP: PSNR=',num2str(performance2),', time=',num2str(t2)])


%Plot Recovered Signals
figure(1); clf;
subplot(1,3,1);
imshow(uint8(x_0));title('Original Image');
subplot(1,3,2);
imshow(uint8(x_hat1));title([denoiser1, '-RED']);
subplot(1,3,3);
imshow(uint8(x_hat2));title([denoiser2, '-AMP']);

