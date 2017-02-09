function [pyr,pind] = buildWUpyr(im, Nsc, daub_order);

% [PYR, INDICES] = buildWUpyr(IM, HEIGHT, DAUB_ORDER)
% 
% Construct a separable undecimated orthonormal QMF/wavelet pyramid
% on matrix (or vector) IM.
% 
% HEIGHT specifies the number of pyramid levels to build. Default
% is maxPyrHt(IM,FILT).  You can also specify 'auto' to use this value.
% 
% DAUB_ORDER: specifies the order of the daubechies wavelet filter used
% 
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband. 

% JPM, Univ. de Granada, 03/2003, based on Rice Wavelet Toolbox 
% function "mrdwt" and on Matlab Pyrtools from Eero Simoncelli.

if Nsc < 1,
    display('Error: Number of scales must be >=1.');
else,   

Nor = 3; % fixed number of orientations;
h = daubcqf(daub_order);
[lpr,yh] = mrdwt(im, h, Nsc+1); % performs the decomposition

[Ny,Nx] = size(im);

% Reorganize the output, forcing the same format as with buildFullSFpyr2

pyr = [];
pind = zeros((Nsc+1)*Nor+2,2);    % Room for a "virtual" high pass residual, for compatibility
nband = 1;
for nsc = 1:Nsc+1,
    for nor = 1:Nor,
        nband = nband + 1;
        band = yh(:,(nband-2)*Nx+1:(nband-1)*Nx);
        sh = (daub_order/2 - 1)*2^nsc;  % approximate phase compensation
        if nor == 1,        % horizontal
            band = shift(band, [sh 2^(nsc-1)]);
        elseif nor == 2,    % vertical
            band = shift(band, [2^(nsc-1) sh]);
        else
            band = shift(band, [sh sh]);    % diagonal
        end    
        if nsc>2,
            band = real(shrink(band,2^(nsc-2)));  % The low freq. bands are shrunk in the freq. domain
        end
        pyr = [pyr; vector(band)];
        pind(nband,:) = size(band);
    end    
end            

band = lpr;
band = shrink(band,2^Nsc);
pyr = [pyr; vector(band)];
pind(nband+1,:) = size(band);


end
