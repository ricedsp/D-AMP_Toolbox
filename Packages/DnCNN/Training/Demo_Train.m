%%% Train the model
addpath('~/matconvnet/matlab');
vl_setupnn

rng('default')

sigma_min_array=40;%[ 0, 10, 20, 40, 60, 80,100,150,300, 500];
sigma_max_array=60;%[10, 20, 40, 60, 80,100,150,300,500,1000];

for i=1:length(sigma_min_array)
    gpuDevice();

    sigma_min = sigma_min_array(i);
    sigma_max = sigma_max_array(i);
    
    opts=struct();

    %%%-------------------------------------------------------------------------
    %%% Configuration
    %%%-------------------------------------------------------------------------
%     opts.modelName        = ['DnCNN_17Layer_sigma',num2str(sigma_min),'to',num2str(sigma_max)]; %%% model name
    opts.modelName        = ['DnCNN_20Layer_sigma',num2str(sigma_min),'to',num2str(sigma_max)]; %%% model name
    opts.learningRate     = 1e-3*ones(1,50);%This will be overwritten adaptively
    % opts.batchSize        = 128;
    opts.batchSize        = 512;
    opts.gpus             = [1]; %%% Indicates whether or not to use a gpu
    
    opts.numSubBatches    = 2;
    opts.bnormLearningRate= 0;
    % opts.bnormLearningRate= 1e-3;
    % opts.bnormLearningRate= 1-1e-3;%bnorm learning rate seems to be used as 1-LR
    opts.conserveMemory = false;
    
    %%% solver
    opts.solver           = 'Adam';
    opts.numberData       = 1;

    %opts.DataDir          = '../../../LDAMP_TensorFlow/TrainingData/multires_traindata_50_128.mat';
    opts.DataDir          = '../../../LDAMP_TensorFlow/TrainingData/tinytraindata_50_128.mat';
    opts.ValDataDir          = '../../../LDAMP_TensorFlow/TrainingData/multires_valdata_50_128.mat';

    opts.gradientClipping = false; %%% set 'true' to prevent exploding gradients in the beginning.
    opts.backPropDepth    = Inf;
    %%%-------------------------------------------------------------------------
    %%%   Initialize model and load data
    %%%-------------------------------------------------------------------------
    %%%  model
    net = DnCNN_init(20);

    %%%  load data
    opts.expDir      = fullfile('NewNetworks', opts.modelName);

    %%%-------------------------------------------------------------------------
    %%%   Train 
    %%%-------------------------------------------------------------------------

    t0=tic;
    [net, info] = DnCNN_train(sigma_min,sigma_max, net,  ...
        'expDir', opts.expDir, ...
        'learningRate',opts.learningRate, ...
        'bnormLearningRate',opts.bnormLearningRate, ...
        'numSubBatches',opts.numSubBatches, ...
        'numberData',opts.numberData, ...
        'backPropDepth',opts.backPropDepth, ...
        'DataDir',opts.DataDir, ...
        'ValDataDir',opts.ValDataDir, ...
        'solver',opts.solver, ...
        'gradientClipping',opts.gradientClipping, ...
        'batchSize', opts.batchSize, ...
        'modelname', opts.modelName, ...
        'gpus',opts.gpus) ;
    training_time=toc(t0);
    save([fullfile('NewNetworks',opts.modelName,[opts.modelName,'_traintime.mat'])],'training_time');
end





