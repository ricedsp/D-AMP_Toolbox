% [PYR, INDICES, STEERMTX, HARMONICS] = buildFullSFpyr2(IM, HEIGHT, ORDER, TWIDTH)
%
% Construct a steerable pyramid on matrix IM, in the Fourier domain.
% Unlike the standard transform, subdivides the highpass band into
% orientations.

function [pyr,pind,steermtx,harmonics] = buildFullSFpyr2(im, ht, order, twidth)

%-----------------------------------------------------------------
%% DEFAULTS:

max_ht = floor(log2(min(size(im)))+1);

if (exist('ht') ~= 1)
  ht = max_ht;
else
  if (ht > max_ht)
    error(sprintf('Cannot build pyramid higher than %d levels.',max_ht));
  end
end

if (exist('order') ~= 1)
  order = 3;
elseif ((order > 15)  | (order < 0))
  fprintf(1,'Warning: ORDER must be an integer in the range [0,15]. Truncating.\n');
  order = min(max(order,0),15);
else
  order = round(order);
end
nbands = order+1;

if (exist('twidth') ~= 1)
  twidth = 1;
elseif (twidth <= 0)
  fprintf(1,'Warning: TWIDTH must be positive.  Setting to 1.\n');
  twidth = 1;
end

%-----------------------------------------------------------------
%% Steering stuff:

if (mod((nbands),2) == 0)
  harmonics = [0:(nbands/2)-1]'*2 + 1;
else
  harmonics = [0:(nbands-1)/2]'*2;
end

steermtx = steer2HarmMtx(harmonics, pi*[0:nbands-1]/nbands, 'even');

%-----------------------------------------------------------------

dims = size(im);
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
lo0mask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
imdft = fftshift(fft2(im));
lo0dft =  imdft .* lo0mask;

[pyr,pind] = buildSFpyrLevs(lo0dft, log_rad, Xrcos, Yrcos, angle, ht, nbands);

%% Split the highpass band into orientations

hi0mask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);

lutsize = 1024;
Xcosn = pi*[-(2*lutsize+1):(lutsize+1)]/lutsize;  % [-2*pi:pi]
order = nbands-1;
const = (2^(2*order))*(factorial(order)^2)/(nbands*factorial(2*order));
Ycosn = sqrt(const) * (cos(Xcosn)).^order;

bands = zeros(prod(size(imdft)), nbands);
bind = zeros(nbands,2);

for b = 1:nbands
    anglemask = pointOp(angle, Ycosn, Xcosn(1)+pi*(b-1)/nbands, Xcosn(2)-Xcosn(1));
    Mask = ((-sqrt(-1))^(nbands-1))*anglemask.*hi0mask;
    % make real the contents in the HF cross (to avoid information loss in these freqs.)
    % It distributes evenly these contents among the nbands orientations
    Mask(1,:) = ones(1,size(im,2))/sqrt(nbands);
    Mask(2:size(im,1),1) = ones(size(im,1)-1,1)/sqrt(nbands);

    banddft =  imdft .* Mask;
    band = real(ifft2(fftshift(banddft)));
    bands(:,b) = real(band(:));
    bind(b,:)  = size(band);
end

pyr = [bands(:); pyr];
pind = [bind; pind];

pind = [ [0 0]; pind];  %% Dummy highpass
