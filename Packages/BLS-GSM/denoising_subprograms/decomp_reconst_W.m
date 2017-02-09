function fh = decomp_reconst_W(im,Nsc,filter,block,noise,parent,covariance,optim,sig);

% Decompose image into subbands, denoise using BLS-GSM method, and recompose again.
%		fh = decomp_reconst(im,Nsc,filter,block,noise,parent,covariance,optim,sig);
%       im:         image
%       Nsc:        number of scales
%       filter:     type of filter used (see namedFilters)
%       block:      2x1 vector indicating the dimensions (rows and columns) of the spatial neighborhood 
%       noise:      signal with the same autocorrelation as the noise
%       parent:     include (1) or not (0) a coefficient from the immediately coarser scale in the neighborhood
%       covariance:	 are we considering covariance or just variance?
%       optim:		 for choosing between BLS-GSM (optim = 1) and MAP-GSM (optim = 0)
%       sig:        standard deviation (scalar for uniform noise or matrix for spatially varying noise)
% Version using a critically sampled pyramid (orthogonal wavelet), as implemented in MatlabPyrTools (Eero).

% JPM, Univ. de Granada, 3/03

if (block(1)/2==floor(block(1)/2))|(block(2)/2==floor(block(2)/2)),
   error('Spatial dimensions of neighborhood must be odd!');
end   

if ~exist('parent'),
        parent = 1;
end

if ~exist('covariance'),
        covariance = 1;
end

if ~exist('optim'),
        optim = 1;
end

if ~exist('sig'),
        sig = sqrt(mean(noise.^2));
end

Nor = 3;    % number of orientations: vertical, horizontal and mixed diagonals (for compatibility)

[pyr,pind] = buildWpyr(im,Nsc,filter,'circular');
[pyrN,pind] = buildWpyr(noise,Nsc,filter,'circular');
pyrh = pyr;
Nband = size(pind,1);
for nband = 1:Nband-1, % everything except the low-pass residual
  fprintf('%d % ',round(100*(nband-1)/(Nband-1)))
  aux = pyrBand(pyr, pind, nband);
  auxn = pyrBand(pyrN, pind, nband);
  prnt = parent & (nband < Nband-Nor);   % has the subband a parent?
  BL = zeros(size(aux,1),size(aux,2),1 + prnt);
  BLn = zeros(size(aux,1),size(aux,2),1 + prnt);
  BL(:,:,1) = aux;
  BLn(:,:,1) = auxn;
  if prnt,
  	aux = pyrBand(pyr, pind, nband+Nor);
    auxn = pyrBand(pyrN, pind, nband+Nor);
    aux = real(expand(aux,2));
    auxn = real(expand(auxn,2));
    BL(:,:,2) = aux;
  	BLn(:,:,2) = auxn;
  end
  
  sy2 = mean2(BL(:,:,1).^2);
  sn2 = mean2(BLn(:,:,1).^2);
  if sy2>sn2,
     SNRin = 10*log10((sy2-sn2)/sn2);
  else
     disp('Signal is not detectable in noisy subband');
  end   
  
  % main
  BL = denoi_BLS_GSM_band(BL,block,BLn,prnt,covariance,optim,sig);
  pyrh(pyrBandIndices(pind,nband)) = BL(:)';
end
fh = reconWpyr(pyrh,pind,filter,'circular');
