function [net, state] = DnCNN_train(sigma_min,sigma_max, net, varargin)

%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option).

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%%%-------------------------------------------------------------------------
%%% solvers: SGD(default) and Adam with(default)/without gradientClipping
%%%-------------------------------------------------------------------------

%%% solver: Adam
%%% opts.solver = 'Adam';
opts.beta1   = 0.9;
opts.beta2   = 0.999;
opts.alpha   = 0.01;
opts.epsilon = 1e-8;


%%% solver: SGD
opts.solver = 'SGD';
opts.learningRate = 0.01;
opts.weightDecay  = 0.001;
opts.momentum     = 0.9 ;

%%% GradientClipping
opts.gradientClipping = false;
opts.theta            = 0.005;

%%% specific parameter for Bnorm
opts.bnormLearningRate = 0;

%%%-------------------------------------------------------------------------
%%%  setting for simplenn
%%%-------------------------------------------------------------------------

opts.conserveMemory = true;
opts.mode = 'normal';
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.skipForward = false;
opts.numSubBatches = 1;
%%%-------------------------------------------------------------------------
%%%  setting for model
%%%-------------------------------------------------------------------------

opts.batchSize = 128 ;
opts.gpus = [];
opts.numEpochs = 300 ;
opts.modelName   = 'model';
opts.expDir = fullfile('data',opts.modelName) ;
opts.numberData   = 1;
opts.DataDir      = '../../../LDAMP_TensorFlow/TrainingData/traindata_50_128.mat';
opts.ValDataDir          = '../../../LDAMP_TensorFlow/TrainingData/valdata_50_128.mat';

%%%-------------------------------------------------------------------------
%%%  update settings
%%%-------------------------------------------------------------------------

opts = vl_argparse(opts, varargin);
opts.numEpochs = numel(opts.learningRate);

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end

%%%-------------------------------------------------------------------------
%%%  Initialization
%%%-------------------------------------------------------------------------

net = vl_simplenn_tidy(net);    %%% fill in some eventually missing values
net.layers{end-1}.precious = 1;
vl_simplenn_display(net, 'batchSize', opts.batchSize) ;

state.getBatch = getBatch ;

%%%-------------------------------------------------------------------------
%%%  Train and Test
%%%-------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf([opts.modelName,'-epoch-%d.mat'], ep));

start = findLastCheckpoint(opts.expDir,opts.modelName) ;
if start >= 1
    fprintf('%s: resuming by loading epoch %d', mfilename, start) ;
    load(modelPath(start), 'net') ;
    net = vl_simplenn_tidy(net) ;
end

%%% load training and validation data

opts.DataPath    = fullfile(opts.DataDir);
TrainData = load(opts.DataPath) ;
opts.train = find(TrainData.set==1);

opts.ValDataPath    = fullfile(opts.ValDataDir);
ValData = load(opts.ValDataPath) ;
opts.test = find(ValData.set==0);

bad_epochs=0;
max_bad_epochs=2;
for epoch = start+1 : opts.numEpochs
    %%% Train for one epoch.
    
    %Stop training if three epochs produced no improvement in validation
    %error
    if bad_epochs>=max_bad_epochs
        if opts.learningRate(epoch-max_bad_epochs)==.001
            load(modelPath(epoch-max_bad_epochs-1), 'net')%load the last best epoch
            save(modelPath(epoch-1), 'net')%save the just-loaded model as the best, in case it needs to be loaded because errors don't improve
            save(fullfile(opts.expDir, sprintf([opts.modelName,'-best_epoch',num2str(epoch-1)])),'net');
            opts.learningRate(epoch:end)=.0001;
            bad_epochs=0;
        elseif opts.learningRate(epoch-max_bad_epochs)==.0001
            load(fullfile(opts.expDir, sprintf([opts.modelName,'-best_epoch',num2str(epoch-max_bad_epochs-1)])),'net');
            opts.learningRate(epoch:end)=.00001;
            bad_epochs=0;
        else
            break;
        end
    end
    
    state.epoch = epoch ;
    state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate)));
    opts.thetaCurrent = opts.theta(min(epoch, numel(opts.theta)));
    if numel(opts.gpus) == 1
        net = vl_simplenn_move(net, 'gpu') ;
    end
    
    if epoch==start+1
        %Determine Initial Validation Error
        tic
        state.test = opts.test(randperm(numel(opts.test))) ; %%% shuffle
        [net, state, min_val_loss] = process_epoch(sigma_min,sigma_max,net, state, ValData, opts, 'test');
        net.layers{end}.class =[];
        toc
    end
    
    %Perform Training
    t0=cputime;
    state.train = opts.train(randperm(numel(opts.train))) ; %%% shuffle
    [net, state, ~] = process_epoch(sigma_min,sigma_max,net, state, TrainData, opts, 'train');
    net.layers{end}.class =[];
    time_taken=cputime-t0;
    
    %Determine Validation Error
    state.test = opts.test(randperm(numel(opts.test))) ; %%% shuffle
    [net, state, loss] = process_epoch(sigma_min,sigma_max,net, state, ValData, opts, 'test');
    net.layers{end}.class =[];
    
    net = vl_simplenn_move(net, 'cpu');
    
    %Save model
    save(modelPath(epoch), 'net')
    
    %Save as best model if validation error is low
    if loss<min_val_loss
        save(fullfile(opts.expDir, sprintf([opts.modelName,'-best_epoch',num2str(epoch)])),'net');
        min_val_loss=loss;
        bad_epochs=0;
    else
        bad_epochs=bad_epochs+1;
    end
end


%%%-------------------------------------------------------------------------
function  [net, state, total_l2_loss] = process_epoch(sigma_min,sigma_max,net, state, Data, opts, mode)
%%%-------------------------------------------------------------------------

if strcmp(mode,'train')
    
    switch opts.solver
        
        case 'SGD' %%% solver: SGD
            for i = 1:numel(net.layers)
                if isfield(net.layers{i}, 'weights')
                    for j = 1:numel(net.layers{i}.weights)
                        state.layers{i}.momentum{j} = 0;
                    end
                end
            end
            
        case 'Adam' %%% solver: Adam
            for i = 1:numel(net.layers)
                if isfield(net.layers{i}, 'weights')
                    for j = 1:numel(net.layers{i}.weights)
                        state.layers{i}.t{j} = 0;
                        state.layers{i}.m{j} = 0;
                        state.layers{i}.v{j} = 0;
                    end
                end
            end
            
    end
    
end


subset = state.(mode) ;
num = 0 ;
res = [];
total_l2_loss=0;
for t=1:opts.batchSize:numel(subset)
    
    for s=1:opts.numSubBatches
        % get this image batch
        batchStart = t + (s-1);
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
        
        [inputs,labels] = state.getBatch(sigma_min,sigma_max,Data, batch) ;
%         truth=inputs-labels;
%         imshow(truth(:,:,1,1),[]);
        
        if numel(opts.gpus) == 1
            inputs = gpuArray(inputs);
            labels = gpuArray(labels);
        end
        
        if strcmp(mode, 'train')
            dzdy = single(1);
            evalMode = 'normal';%%% forward and backward (Gradients)
        else
            dzdy = [] ;
            evalMode = 'test';  %%% forward only
%             evalMode = 'normal';  %%% forward and backward (Gradients)%Forces the network not to remove batch normalization.
        end
        
        net.layers{end}.class = labels ;
        res = vl_simplenn(net, inputs, dzdy, res, ...
            'accumulate', s ~= 1, ...
            'mode', evalMode, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'cudnn', opts.cudnn) ;
        
    end
    
    if strcmp(mode, 'train')
        [state, net] = params_updates(state, net, res, opts, opts.batchSize) ;
    end
    
    lossL2 = gather(res(end).x) ;
    %%%--------add your code here------------------------
    if mod(floor(t/opts.batchSize),60)==1%show figure every 20 batches
        k=2;
        label_hat=res(end-1).x;
        label_hat_k=squeeze(label_hat(:,:,1,k));
        label_k=squeeze(labels(:,:,1,k));
        sig_hat_k=squeeze(inputs(:,:,1,k))-label_hat_k;
        sig_k=squeeze(inputs(:,:,1,k))-label_k;
        figure(1);
        subplot(3,2,1);imshow(inputs(:,:,1,k),[]);title('Noisy Input');
        subplot(3,2,3);imshow(label_hat_k,[]);title('Noise Estimate');
        subplot(3,2,4);imshow(label_k,[]);title('Noise Truth');
        subplot(3,2,5);imshow(sig_hat_k,[]);title('Signal Estimate');
        subplot(3,2,6);imshow(sig_k,[]);title('Signal Truth');
        Signals=inputs-labels;
        Signal_Estimates=inputs-res(end-1).x;
        MyLossL2=vl_nnloss(Signals,Signal_Estimates);%confirm this value is the same as lossL2
    end
    
    %%%--------------------------------------------------
    
    fprintf('%s: epoch %02d dataset %02d: %3d/%3d:', mode, state.epoch, mod(state.epoch,opts.numberData), ...
        fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    fprintf('error: %f \n', lossL2) ;
    total_l2_loss=total_l2_loss+lossL2;
end


%%%-------------------------------------------------------------------------
function [state, net] = params_updates(state, net, res, opts, batchSize)
%%%-------------------------------------------------------------------------

switch opts.solver
    
    case 'SGD' %%% solver: SGD
        
        for l=numel(net.layers):-1:1
            for j=1:numel(res(l).dzdw)
                if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
                    %%% special case for learning bnorm moments
                    thisLR = net.layers{l}.learningRate(j) - opts.bnormLearningRate;
                    net.layers{l}.weights{j} = vl_taccum(...
                        1 - thisLR, ...
                        net.layers{l}.weights{j}, ...
                        thisLR / batchSize, ...
                        res(l).dzdw{j}) ;
                    
                else
                    thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j);
                    thisLR = state.learningRate * net.layers{l}.learningRate(j);
                    
                    if opts.gradientClipping
                        theta = opts.thetaCurrent/thisLR;
                        state.layers{l}.momentum{j} = opts.momentum * state.layers{l}.momentum{j} ...
                            - thisDecay * net.layers{l}.weights{j} ...
                            - (1 / batchSize) * gradientClipping(res(l).dzdw{j},theta) ;
                        net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
                            thisLR * state.layers{l}.momentum{j} ;
                    else
                        state.layers{l}.momentum{j} = opts.momentum * state.layers{l}.momentum{j} ...
                            - thisDecay * net.layers{l}.weights{j} ...
                            - (1 / batchSize) * res(l).dzdw{j} ;
                        net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
                            thisLR * state.layers{l}.momentum{j} ;
                    end
                end
            end
        end
        
        
    case 'Adam'  %%% solver: Adam
        
        for l=numel(net.layers):-1:1
            for j=1:numel(res(l).dzdw)
                
                if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
                    
                    %%% special case for learning bnorm moments
                    thisLR = net.layers{l}.learningRate(j) - opts.bnormLearningRate;
                    net.layers{l}.weights{j} = vl_taccum(...
                        1 - thisLR, ...
                        net.layers{l}.weights{j}, ...
                        thisLR / batchSize, ...
                        res(l).dzdw{j}) ;
                else
                    thisLR = state.learningRate * net.layers{l}.learningRate(j);
                    state.layers{l}.t{j} = state.layers{l}.t{j} + 1;
                    t = state.layers{l}.t{j};
                    alpha = thisLR;
                    lr = alpha * sqrt(1 - opts.beta2^t) / (1 - opts.beta1^t);
                    
                    state.layers{l}.m{j} = state.layers{l}.m{j} + (1 - opts.beta1) .* (res(l).dzdw{j} - state.layers{l}.m{j});
                    state.layers{l}.v{j} = state.layers{l}.v{j} + (1 - opts.beta2) .* (res(l).dzdw{j} .* res(l).dzdw{j} - state.layers{l}.v{j});
                    
                    if opts.gradientClipping
                        theta = opts.thetaCurrent/lr;
                        net.layers{l}.weights{j} = net.layers{l}.weights{j} - lr * gradientClipping(state.layers{l}.m{j} ./ (sqrt(state.layers{l}.v{j}) + opts.epsilon),theta);
                    else
                        net.layers{l}.weights{j} = net.layers{l}.weights{j} - lr * state.layers{l}.m{j} ./ (sqrt(state.layers{l}.v{j}) + opts.epsilon);
                    end
                    
                    % net.layers{l}.weights{j} = weightClipping(net.layers{l}.weights{j},2); % gradually clip the weights
                    
                end
            end
        end
end


%%%-------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir,modelName)
%%%-------------------------------------------------------------------------
list = dir(fullfile(modelDir, [modelName,'-epoch-*.mat'])) ;
tokens = regexp({list.name}, [modelName,'-epoch-([\d]+).mat'], 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

%%%-------------------------------------------------------------------------
function A = gradientClipping(A, theta)
%%%-------------------------------------------------------------------------
A(A>theta)  = theta;
A(A<-theta) = -theta;

%%%-------------------------------------------------------------------------
function A = weightClipping(A, theta)
%%%-------------------------------------------------------------------------
A(A>theta)  = A(A>theta) -0.0005;
A(A<-theta) = A(A<-theta)+0.0005;


%%%-------------------------------------------------------------------------
function fn = getBatch
%%%-------------------------------------------------------------------------
fn = @(sigma_min,sigma_max,x,y) getSimpleNNBatch(sigma_min,sigma_max,x,y);

%%%-------------------------------------------------------------------------
function [inputs,labels] = getSimpleNNBatch(sigma_min,sigma_max,Data, batch)
%%%-------------------------------------------------------------------------
inputs = Data.inputs(:,:,:,batch);
rng('shuffle');
mode = randperm(8);
inputs = data_augmentation(inputs, mode(1));
sigma_vec=sigma_min+rand(numel(batch),1)*(sigma_max-sigma_min);%amount of random noise to apply to each image
sigma_array=ones(size(inputs));
for i=1:numel(batch)
    sigma_array(:,:,1,i)=sigma_vec(i);
end
labels  = sigma_array/255.*randn(size(inputs),'single');
inputs = inputs + labels;

function image = data_augmentation(image, mode)

if mode == 1
    return;
end

if mode == 2 % flipped
    image = flipud(image);
    return;
end

if mode == 3 % rotation 90
    image = rot90(image,1);
    return;
end

if mode == 4 % rotation 90 & flipped
    image = rot90(image,1);
    image = flipud(image);
    return;
end

if mode == 5 % rotation 180
    image = rot90(image,2);
    return;
end

if mode == 6 % rotation 180 & flipped
    image = rot90(image,2);
    image = flipud(image);
    return;
end

if mode == 7 % rotation 270
    image = rot90(image,3);
    return;
end

if mode == 8 % rotation 270 & flipped
    image = rot90(image,3);
    image = flipud(image);
    return;
end







