function fh = decomp_reconst(fn,Nsc,Nor,block,noise,parent,covariance,optim,sig);

% Decompose image into subbands, denoise, and recompose again.
%	fh = decomp_reconst(fn,Nsc,Nor,block,noise,parent);

% Javier Portilla, Univ. de Granada, 5/02
% Last revision: 11/04

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

[pyr,pind] = buildSFpyr(fn,Nsc,Nor-1);
[pyrN,pind] = buildSFpyr(noise,Nsc,Nor-1);
pyrh = real(pyr);
Nband = size(pind,1);
for nband = 1:Nband -1,
  fprintf('%d % ',round(100*nband/(Nband-1)))
  aux = pyrBand(pyr, pind, nband);
  auxn = pyrBand(pyrN, pind, nband);
  [Nsy,Nsx] = size(aux);  
  prnt = parent & (nband < Nband-1-Nor) & (nband>1);
  BL = zeros(size(aux,1),size(aux,2),1 + prnt);
  BLn = zeros(size(aux,1),size(aux,2),1 + prnt);
  BL(:,:,1) = aux;
  BLn(:,:,1) = auxn*sqrt(((Nsy-2)*(Nsx-2))/(Nsy*Nsx));     % because we are discarding 2 coefficients on every dimension  
  if prnt,
  	aux = pyrBand(pyr, pind, nband+Nor);
	aux = real(expand(aux,2));
  	auxn = pyrBand(pyrN, pind, nband+Nor);
	auxn = real(expand(auxn,2));
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
fh = reconSFpyr(pyrh,pind);
