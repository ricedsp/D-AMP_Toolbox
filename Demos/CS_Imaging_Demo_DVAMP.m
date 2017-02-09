%Demonstrates compressively sampling and D-VAMP recovery of an image.

%rng(1)
addpath(genpath('..'));

%Parameters
%Available denoiser options are NLM, Gauss, Bilateral, BLS-GSM, BM3D, fast-BM3D, and BM3D-SAPCA 
%Note: BM3D-SAPCA requires Matlab 2015a or earlier and is extremely slow!
denoiser1='BM3D';
denoiser2='fast-BM3D';
filename='barbara.png';
SamplingRate=.2;
itersAMP=30;
itersVAMP=30;
imsize=128; 

ImIn=double(imread(filename));
x_0=imresize(ImIn,imsize/size(ImIn,1));
[height, width]=size(x_0);
n=length(x_0(:));
m=round(n*SamplingRate);
errfxn = @(x_hat) PSNR(x_0,reshape(x_hat,[height width]));

%Generate Gaussian Measurement Matrix
M=randn(m,n);
for j = 1:n
    M(:,j) = M(:,j) ./ sqrt(sum(abs(M(:,j)).^2));
end

%Compressively sample the image
y=M*x_0(:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Recover Signal using D-AMP algorithms %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tstart1 = tic;
[x_hat1,psnr1] = DAMP(y,itersAMP,height,width,denoiser1,M,[],errfxn);
time1 = toc(tstart1);
performance1=PSNR(x_0,x_hat1);
fprintf('\n');
display([num2str(SamplingRate*100),'pct Sampling ',denoiser1,'-AMP Reconstruction PSNR=',num2str(performance1),', time=',num2str(time1)])

tstart2 = tic;
[x_hat2,psnr2] = DAMP(y,itersAMP,height,width,denoiser2,M,[],errfxn);
time2 = toc(tstart2);
performance2=PSNR(x_0,x_hat2);
fprintf('\n');
display([num2str(SamplingRate*100),'pct Sampling ',denoiser2,'-AMP Reconstruction PSNR=',num2str(performance2),', time=',num2str(time2)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Recover Signal using D-VAMP algorithms %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

external_SVD = true; 
if external_SVD
  [U,D] = eig(M*M');
  Mt = @(z) M'*z;
  M = @(x) M*x;
  Ut = @(z) U'*z;
  U = @(x) U*x;
  d = diag(D);
else
  Mt = [];
  U = [];
  Ut = [];
  d = [];
end

tstart3 = tic;
[x_hat3,psnr3] = DVAMP(y,itersVAMP,height,width,denoiser1,M,Mt,errfxn,U,Ut,d);
time3 = toc(tstart3);
performance3=PSNR(x_0,x_hat3);
fprintf('\n');
display([num2str(SamplingRate*100),'pct Sampling ',denoiser1,'-VAMP Reconstruction PSNR=',num2str(performance3),', time=',num2str(time3)])

tstart4 = tic;
[x_hat4,psnr4] = DVAMP(y,itersVAMP,height,width,denoiser2,M,Mt,errfxn,U,Ut,d);
time4 = toc(tstart4);
performance4=PSNR(x_0,x_hat4);
fprintf('\n');
display([num2str(SamplingRate*100),'pct Sampling ',denoiser2,'-VAMP Reconstruction PSNR=',num2str(performance4),', time=',num2str(time4)])

%%%%%%%%%%%%%%%%%%%%%%%%
%Plot Recovered Signals%
%%%%%%%%%%%%%%%%%%%%%%%%

figure(1); clf;
subplot(2,3,1);
imshow(uint8(x_0));title('Original Image');
subplot(2,3,2);
imshow(uint8(x_hat1));title([denoiser1, '-AMP']);
subplot(2,3,3);
imshow(uint8(x_hat2));title([denoiser2, '-AMP']);
subplot(2,3,5);
imshow(uint8(x_hat3));title([denoiser1, '-VAMP']);
subplot(2,3,6);
imshow(uint8(x_hat4));title([denoiser2, '-VAMP']);

%%%%%%%%%%%%%%%%%%%%%%%%
%Plot PSNR Trajectories%
%%%%%%%%%%%%%%%%%%%%%%%%

figure(2); clf;
plot(psnr1,'.-','Displayname',[denoiser1,'-AMP'])
hold on; 
plot(psnr2,'.-','Displayname',[denoiser2,'-AMP']); 
plot(psnr3,'.-','Displayname',[denoiser1,'-VAMP']); 
plot(psnr4,'.-','Displayname',[denoiser2,'-VAMP']); 
hold off;
grid on;
legend(gca,'Show','Location','SouthEast')
xlabel('iteration')
ylabel('PSNR')
