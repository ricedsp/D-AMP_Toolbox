function fh = decomp_reconst_full(im,Nsc,Nor,block,noise,parent,covariance,optim,sig);

% Decompose image into subbands, denoise, and recompose again.
%		fh = decomp_reconst(im,Nsc,Nor,block,noise,parent,covariance,optim,sig);
%       covariance:	 are we considering covariance or just variance?
%       optim:		 for choosing between BLS-GSM (optim = 1) and MAP-GSM (optim = 0)
%       sig:        standard deviation (scalar for uniform noise or matrix for spatially varying noise)
% Version using the Full steerable pyramid (2) (High pass residual
% splitted into orientations).

% JPM, Univ. de Granada, 5/02
% Last Revision: 11/04

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

[pyr,pind] = buildFullSFpyr2(im,Nsc,Nor-1);
[pyrN,pind] = buildFullSFpyr2(noise,Nsc,Nor-1);
pyrh = real(pyr);
Nband = size(pind,1)-1;
for nband = 2:Nband, % everything except the low-pass residual
  fprintf('%d % ',round(100*(nband-1)/(Nband-1)))
  aux = pyrBand(pyr, pind, nband);
  auxn = pyrBand(pyrN, pind, nband);
  [Nsy,Nsx] = size(aux);
  prnt = parent & (nband < Nband-Nor);   % has the subband a parent?
  BL = zeros(size(aux,1),size(aux,2),1 + prnt);
  BLn = zeros(size(aux,1),size(aux,2),1 + prnt);
  BL(:,:,1) = aux;
  BLn(:,:,1) = auxn*sqrt(((Nsy-2)*(Nsx-2))/(Nsy*Nsx));     % because we are discarding 2 coefficients on every dimension  
  if prnt,
  	aux = pyrBand(pyr, pind, nband+Nor);
    auxn = pyrBand(pyrN, pind, nband+Nor);
    if nband>Nor+1,     % resample 2x2 the parent if not in the high-pass oriented subbands.
	   aux = real(expand(aux,2));
       auxn = real(expand(auxn,2));
    end    
  	BL(:,:,2) = aux;
    BLn(:,:,2) = auxn*sqrt(((Nsy-2)*(Nsx-2))/(Nsy*Nsx)); % because we are discarding 2 coefficients on every dimension       
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
fh = reconFullSFpyr2(pyrh,pind);
