function res = reconWUpyr(pyr, pind, daub_order);

% RES = reconWUpyr(PYR, INDICES, DAUB_ORDER)
 
% Reconstruct image from its separable undecimated orthonormal QMF/wavelet pyramid
% representation, as created by buildWUpyr.
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  
% 
% DAUB_ORDER: specifies the order of the daubechies wavelet filter used
 
% JPM, Univ. de Granada, 03/2003, based on Rice Wavelet Toolbox 
% functions "mrdwt" and  "mirdwt", and on Matlab Pyrtools from Eero Simoncelli.


Nor = 3;
Nsc = (size(pind,1)-2)/Nor-1;
h = daubcqf(daub_order);

yh = [];

nband = 1;
last = prod(pind(1,:)); % empty "high pass residual band" for compatibility with full steerpyr 2
for nsc = 1:Nsc+1,  % The number of scales corresponds to the number of pyramid levels (also for compatibility)
    for nor = 1:Nor,
        nband = nband +1;
        first = last + 1;
        last = first + prod(pind(nband,:)) - 1;
        band = pyrBand(pyr,pind,nband);
        sh = (daub_order/2 - 1)*2^nsc;  % approximate phase compensation
        if nsc > 2,
            band = expand(band, 2^(nsc-2));
        end   
        if nor == 1,        % horizontal
            band = shift(band, [-sh -2^(nsc-1)]);
        elseif nor == 2,    % vertical
            band = shift(band, [-2^(nsc-1) -sh]);
        else
            band = shift(band, [-sh -sh]);    % diagonal
        end    
        yh = [yh band];    
    end    
end    

nband = nband + 1;
band = pyrBand(pyr,pind,nband);
lpr = expand(band,2^Nsc);

res= mirdwt(lpr,yh,h,Nsc+1);