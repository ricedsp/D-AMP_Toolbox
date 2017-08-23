function [] = LoadNetworkWeights(n_layers)
    %Load the shared NN-based denoisers network
    if ~exist('vl_setupnn','file')
        error('vl_setupnn not found. Make sure matconvnet is in your path');
    end
    vl_setupnn
    global net_0to10
    global net_10to20
    global net_20to40
    global net_40to60
    global net_60to80
    global net_80to100
    global net_100to150
    global net_150to300
    global net_300to500
    global net_500to1000
    if nargin~=0
        if n_layers==17
            %Load 17 layer network weights
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma0to10-best');%loads net
            net.layers = net.layers(1:end-1);
            net_0to10 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma10to20-best');%loads net
            net.layers = net.layers(1:end-1);
            net_10to20 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma20to40-best');%loads net
            net.layers = net.layers(1:end-1);
            net_20to40 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma40to60-best');%loads net
            net.layers = net.layers(1:end-1);
            net_40to60 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma60to80-best');%loads net
            net.layers = net.layers(1:end-1);
            net_60to80 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma80to100-best');%loads net
            net.layers = net.layers(1:end-1);
            net_80to100 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma100to150-best');%loads net
            net.layers = net.layers(1:end-1);
            net_100to150 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma150to300-best');%loads net
            net.layers = net.layers(1:end-1);
            net_150to300 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma300to500-best');%loads net
            net.layers = net.layers(1:end-1);
            net_300to500 = vl_simplenn_move(net, 'gpu') ;
            load('./../Packages/DnCNN/BestNets_17/DnCNN_17Layer_sigma500to1000-best');%loads net
            net.layers = net.layers(1:end-1);
            net_500to1000 = vl_simplenn_move(net, 'gpu') ;
        else
            LoadNetworkWeights();
        end
    else
        %Load 20 layer network weights
        load('./../Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma0to10-best');%loads net
        net.layers = net.layers(1:end-1);
        net_0to10 = vl_simplenn_move(net, 'gpu') ;
        load('./../Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma10to20-best');%loads net
        net.layers = net.layers(1:end-1);
        net_10to20 = vl_simplenn_move(net, 'gpu') ;
        load('./../Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma20to40-best');%loads net
        net.layers = net.layers(1:end-1);
        net_20to40 = vl_simplenn_move(net, 'gpu') ;
        load('./../Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma40to60-best');%loads net
        net.layers = net.layers(1:end-1);
        net_40to60 = vl_simplenn_move(net, 'gpu') ;
        load('./../Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma60to80-best');%loads net
        net.layers = net.layers(1:end-1);
        net_60to80 = vl_simplenn_move(net, 'gpu') ;
        load('./../Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma80to100-best');%loads net
        net.layers = net.layers(1:end-1);
        net_80to100 = vl_simplenn_move(net, 'gpu') ;
        load('./../Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma100to150-best');%loads net
        net.layers = net.layers(1:end-1);
        net_100to150 = vl_simplenn_move(net, 'gpu') ;
        load('./../Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma150to300-best');%loads net
        net.layers = net.layers(1:end-1);
        net_150to300 = vl_simplenn_move(net, 'gpu') ;
        load('./../Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma300to500-best');%loads net
        net.layers = net.layers(1:end-1);
        net_300to500 = vl_simplenn_move(net, 'gpu') ;
        load('./../Packages/DnCNN/BestNets_20/DnCNN_20Layer_sigma500to1000-best');%loads net
        net.layers = net.layers(1:end-1);
        net_500to1000 = vl_simplenn_move(net, 'gpu') ;
    end
end

