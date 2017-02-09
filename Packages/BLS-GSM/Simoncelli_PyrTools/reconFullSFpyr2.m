% RES = reconFullSFpyr2(PYR, INDICES, LEVS, BANDS, TWIDTH)
%
% Reconstruct image from its steerable pyramid representation, in the Fourier
% domain, as created by buildSFpyr.
% Unlike the standard transform, subdivides the highpass band into
% orientations.


function res = reconFullSFpyr2(pyr, pind, levs, bands, twidth)

%%------------------------------------------------------------
%% DEFAULTS:

if (exist('levs') ~= 1)
  levs = 'all';
end

if (exist('bands') ~= 1)
  bands = 'all';
end

if (exist('twidth') ~= 1)
  twidth = 1;
elseif (twidth <= 0)
  fprintf(1,'Warning: TWIDTH must be positive.  Setting to 1.\n');
  twidth = 1;
end

%%------------------------------------------------------------

nbands = spyrNumBands(pind)/2;

maxLev =  2+spyrHt(pind(nbands+1:size(pind,1),:));
if strcmp(levs,'all')
  levs = [0:maxLev]';
else
  if (any(levs > maxLev) | any(levs < 0))
    error(sprintf('Level numbers must be in the range [0, %d].', maxLev));
  end
  levs = levs(:);
end

if strcmp(bands,'all')
  bands = [1:nbands]';
else
  if (any(bands < 1) | any(bands > nbands))
    error(sprintf('Band numbers must be in the range [1,3].', nbands));
  end
  bands = bands(:);
end

%----------------------------------------------------------------------

dims = pind(2,:);
ctr = ceil((dims+0.5)/2);

[xramp,yramp] = meshgrid( ([1:dims(2)]-ctr(2))./(dims(2)/2), ...
    ([1:dims(1)]-ctr(1))./(dims(1)/2) );
angle = atan2(yramp,xramp);
log_rad = sqrt(xramp.^2 + yramp.^2);
log_rad(ctr(1),ctr(2)) =  log_rad(ctr(1),ctr(2)-1);
log_rad  = log2(log_rad);

%% Radial transition function (a raised cosine in log-frequency):
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Yrcos = sqrt(Yrcos);
YIrcos = sqrt(1.0 - Yrcos.^2);

if (size(pind,1) == 2)
  if (any(levs==1))
    resdft = fftshift(fft2(pyrBand(pyr,pind,2)));
  else
    resdft = zeros(pind(2,:));
  end
else
  resdft = reconSFpyrLevs(pyr(1+sum(prod(pind(1:nbands+1,:)')):size(pyr,1)), ...
      pind(nbands+2:size(pind,1),:), ...
      log_rad, Xrcos, Yrcos, angle, nbands, levs, bands);
end
 
lo0mask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
resdft = resdft .* lo0mask;

%% Oriented highpass bands:
if any(levs == 0)
  lutsize = 1024;
  Xcosn = pi*[-(2*lutsize+1):(lutsize+1)]/lutsize;  % [-2*pi:pi]
  order = nbands-1;
  %% divide by sqrt(sum_(n=0)^(N-1)  cos(pi*n/N)^(2(N-1)) )
  const = (2^(2*order))*(factorial(order)^2)/(nbands*factorial(2*order));
  Ycosn = sqrt(const) * (cos(Xcosn)).^order;

  hi0mask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);

  ind = 1;
  for b = 1:nbands
    if any(bands == b)
      anglemask = pointOp(angle,Ycosn,Xcosn(1)+pi*(b-1)/nbands,Xcosn(2)-Xcosn(1));
      band = reshape(pyr(ind:ind+prod(dims)-1), dims(1), dims(2));
      banddft = fftshift(fft2(band));
      % make real the contents in the HF cross (to avoid information loss in these freqs.)
      % It distributes evenly these contents among the nbands orientations
      Mask = (sqrt(-1))^(nbands-1) * anglemask.*hi0mask;
      Mask(1,:) = ones(1,size(Mask,2))/sqrt(nbands);
      Mask(2:size(Mask,1),1) = ones(size(Mask,1)-1,1)/sqrt(nbands);
      
      resdft = resdft + banddft.*Mask;
    end
    ind = ind + prod(dims);
  end
end

res = real(ifft2(ifftshift(resdft)));
