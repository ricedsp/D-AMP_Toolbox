function J = NLMF2Dtree(I, Options)
% This function NLMF2Dtree performs Non-Local Means noise filtering of 
% 2D grey/color image. This function constructs a KD-tree of the whole
% image to find patches which have the closest intensity difference,
% instead of searching only in the local neighboorhood like in the 
% NLMF function.
% Warning, despite the usage of a kd-tree, this function will take more 
% then a hour for one image.
%
% Function:
%
%   J = NLMF2Dtree(I, Options);
% 
% inputs, 
%   I : 2D grey/color or 3D image data, of type Single or Double 
%           in range [0..1]
%   Options : Struct with options
%
% outputs,
%   J : The NL-means filtered image or image volume 
%
% options,
%   Options.kernelratio : Radius of local Patch (default 3)
%   Options.filterstrength : Strength of the NLMF filtering (default 0.05)
%   Options.number : The number of patches which have the closest distance
%                      to a local patch used for filtering (default 9).
%
% First Compile c-code!!!!, with :
%   mex image2vectors_double.c -v
%
% Note: This function uses the "Matlab Statistics Toolbox"
%
% Example 2D color, 
%  I=im2double(imread('lena.jpg'));
%  I=imnoise(I,'gaussian',0.01);
%  Options.kernelratio=4;
%  Options.filterstrength=0.07;
%  J=NLMF2Dtree(I,Options);
%  figure,
%  subplot(1,2,1),imshow(I); title('Noisy image')
%  subplot(1,2,2),imshow(J); title('NL-means image');
%
% See also NLMF, createns, knnsearch.
%
% Function is written by D.Kroon University of Twente (September 2010)

% Process inputs
defaultoptions=struct('kernelratio',3,'filterstrength',0.05,'number',9);
if(~exist('Options','var')), Options=defaultoptions;
else
    tags = fieldnames(defaultoptions);
    for i=1:length(tags), if(~isfield(Options,tags{i})), Options.(tags{i})=defaultoptions.(tags{i}); end, end
    if(length(tags)~=length(fieldnames(Options))),
        warning('NLMF2Dtree:unknownoption','unknown options found');
    end
end

% Detect if this is a color or grey scale image
switch(size(I,3))
    case 1, rgb=false;
    case 3, rgb=true;
    otherwise,error('NLMF2Dtree:inputs','This is not a 2D image');
end

% Pad the image, to allow simple extraction of local patches at the image
% boundary
Ipad = padarray(I,[Options.kernelratio Options.kernelratio],'symmetric'); 

% Get the local patches
V=image2vectors_double(double(Ipad),Options.kernelratio,2);

% Create a KDtree with all local patches
ns = createns(V','nsmethod','kdtree');

% Find the patches closest by in intensity in relation to the local patch
% itself
[VM,D] = knnsearch(ns,V','k',Options.number+1);

% Calculate the weight of the patches used for a pixel
W=exp(-(D.^2)*Options.filterstrength);
% The first column is the local patch itself, set the weight to maximum
% found in the other patches
W(:,1)=max(W(:,2:end),[],2);
% Make the sum of weights which will be used for one patch, equal to one.
W=W./repmat((sum(W,2)+eps),[1 size(W,2)]);

% Add the weighted intensities, of the center pixels of the in intensity
% closest by patches
if(rgb)
    Ir=I(:,:,1);
    Ig=I(:,:,2);
    Ib=I(:,:,3);
    Jr=sum(Ir(VM).*W,2);
    Jg=sum(Ig(VM).*W,2);
    Jb=sum(Ib(VM).*W,2);
    Jr=reshape(Jr,[size(I,1) size(I,2)]);
    Jg=reshape(Jg,[size(I,1) size(I,2)]);
    Jb=reshape(Jb,[size(I,1) size(I,2)]);
    J(:,:,1)=Jr;
    J(:,:,2)=Jg;
    J(:,:,3)=Jb;
else
    J=sum(I(VM).*W,2); 
    J=reshape(J,size(I));
end

