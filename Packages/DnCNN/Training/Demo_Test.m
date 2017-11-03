%%% Test the model
addpath('~/matconvnet/matlab');
addpath('../../../Utils')
addpath('../../../TestImages')
vl_setupnn




%Parameters
sigma_true=50;
sigma_min = 40;
sigma_max = 60;
showResult  = 1;
useGPU      = 1;
pauseTime   = 0;

%Select which network to use
modelName = ['DnCNN_20Layer_sigma',num2str(sigma_min),'to',num2str(sigma_max)]; %%% model name
epoch       = 1;


%%% load Gaussian denoising model
load(fullfile('NewNetworks',modelName,[modelName,'-epoch-',num2str(epoch),'.mat']));
net = vl_simplenn_tidy(net);
net.layers = net.layers(1:end-1);

%%%
net = vl_simplenn_tidy(net);

%%% move to gpu
if useGPU
    net = vl_simplenn_move(net, 'gpu') ;
end

%Create the noisy image
clean=double(imread('barbara.png'));

noisy=clean+40*randn(size(clean));
noisy=single(noisy);

%%% convert to GPU
if useGPU
    noisy = gpuArray(noisy);
end

res    = vl_simplenn(net,noisy,[],[],'conserveMemory',true,'mode','test');
denoised = noisy - res(end).x;

%%% convert to CPU
if useGPU
    denoised = gather(denoised);
    noisy  = gather(noisy);
end

%%% calculate PSNR and SSIM
Input_PSNR=PSNR(denoised,clean);
Output_PSNR=PSNR(noisy,clean);

figure(1);
subplot(1,2,1);imshow(noisy,[]);title('Noisy input');
subplot(1,2,2);imshow(denoised,[]);title('Denoised output');

display('Input psnr: ',Input_PSNR);
display('Output psnr: ',Output_PSNR);




