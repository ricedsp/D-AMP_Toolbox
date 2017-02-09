function im_d = denoi_BLS_GSM(im, sig, PS, blSize, parent, boundary, Nsc, Nor, covariance, optim, repres1, repres2, seed);

% [im_d,im,SNR_N,SNR,PSNR] = denoi_BLS_GSM(im, sig, ft, PS, blSize, parent, boundary, Nsc, Nor, covariance, optim, repres1, repres2, seed);
%
%	im:	input noisy image
%	sig:	standard deviation of noise
%	PS:	Power Spectral Density of noise ( fft2(autocorrelation) )
%			NOTE: scale factors do not matter. Default is white.
%	blSize: 2x1 or 1x2 vector indicating local neighborhood
%		([sY sX], default is [3 3])
%   parent: Use parent yes/no (1/0)
%	Nsc:	Number of scales
%   Nor:  Number of orientations. For separable wavelets this MUST be 3.
%   covariance: Include / Not Include covariance in the GSM model (1/0)
%   optim: BLS / MAP-Wiener(2-step) (1/0)
%   repres1: Possible choices for representation:
%           'w':    orthogonal wavelet
%                   (uses buildWpyr, reconWpyr)
%                   repres2 (optional):
%                      haar:              - Haar wavelet.
%                      qmf8, qmf12, qmf16 - Symmetric Quadrature Mirror Filters [Johnston80]
%                      daub2,daub3,daub4  - Daubechies wavelet [Daubechies88] (#coef = 2N, para daubN).
%                      qmf5, qmf9, qmf13: - Symmetric Quadrature Mirror Filters [Simoncelli88,Simoncelli90]
%           'uw':   undecimated orthogonal wavelet, Daubechies, pyramidal version 
%                   (uses buildWUpyr, reconWUpyr).
%                   repres2 (optional): 'daub<N>', where <N> is a positive integer (e.g., 2)
%           's':    steerable pyramid [Simoncelli&Freeman95].
%                   (uses buildSFpyr, reconSFpyr)
%           'fs':   full steerable pyramid [Portilla&Simoncelli02].
%                   (uses buildFullSFpyr2, reconsFullSFpyr2)       
%   seed (optional):    Seed used for generating the Gaussian noise (when ft == 0)
%                       By default is 0.
%
%   im_d: denoising result

% Javier Portilla, Univ. de Granada, 5/02
% revision 31/03/2003
% revision 7/01/2004
% Last revision 15/11/2004

if ~exist('blSize'),
	blSzX = 3;	% Block size
	blSzY = 3;
else
	blSzY = blSize(1);
	blSzX = blSize(2);
end

if (blSzY/2==floor(blSzY/2))|(blSzX/2==floor(blSzX/2)),
   error('Spatial dimensions of neighborhood must be odd!');
end   

if ~exist('PS'),
    no_white = 0;   % Power spectral density of noise. Default is white noise
else
	no_white = 1;
end

if ~exist('parent'),
	parent = 1;
end

if ~exist('boundary'),
	boundary = 1;
end

if ~exist('Nsc'),
	Nsc = 4;
end

if ~exist('Nor'),
	Nor = 8;
end

if ~exist('covariance'),
        covariance = 1;
end

if ~exist('optim'),
        optim = 1;
end

if ~exist('repres1'),
        repres1 = 'fs';
end

if ~exist('repres2'),
        repres2 = '';
end

if (((repres1=='w') | (repres1=='uw')) & (Nor~=3)),
    warning('For X-Y separable representations Nor must be 3. Nor = 3 assumed.');
    Nor = 3;
end    

if ~exist('seed'),
        seed = 0;
end

[Ny Nx] = size(im);

% We ensure that the processed image has dimensions that are integer
% multiples of 2^(Nsc+1), so it will not crash when applying the 
% pyramidal representation. The idea is padding with mirror reflected
% pixels (thanks to Jesus Malo for this idea).

Npy = ceil(Ny/2^(Nsc+1))*2^(Nsc+1);
Npx = ceil(Nx/2^(Nsc+1))*2^(Nsc+1);

if Npy~=Ny | Npx~=Nx,
    Bpy = Npy-Ny;
    Bpx = Npx-Nx;
    im = bound_extension(im,Bpy,Bpx,'mirror');
    im = im(Bpy+1:end,Bpx+1:end);	% add stripes only up and right
end   


% size of the extension for boundary handling
if (repres1 == 's') | (repres1 == 'fs'),
    By = (blSzY-1)*2^(Nsc-2);
    Bx = (blSzX-1)*2^(Nsc-2);
else        
    By = (blSzY-1)*2^(Nsc-1);
    Bx = (blSzX-1)*2^(Nsc-1);
end    

if ~no_white,       % White noise
    PS = ones(size(im));
end

% As the dimensions of the power spectral density (PS) support and that of the
% image (im) do not need to be the same, we have to adapt the first to the
% second (zero padding and/or cropping).

PS = fftshift(PS);
isoddPS_y = (size(PS,1)~=2*(floor(size(PS,1)/2)));
isoddPS_x = (size(PS,2)~=2*(floor(size(PS,2)/2)));
PS = PS(1:end-isoddPS_y, 1:end-isoddPS_x);          % ensures even dimensions for the power spectrum
PS = fftshift(PS);

[Ndy,Ndx] = size(PS);   % dimensions are even

delta = real(ifft2(sqrt(PS)));      
delta = fftshift(delta);
aux = delta;
delta = zeros(Npy,Npx);
if (Ndy<=Npy)&(Ndx<=Npx),
    delta(Npy/2+1-Ndy/2:Npy/2+Ndy/2,Npx/2+1-Ndx/2:Npx/2+Ndx/2) = aux;
elseif (Ndy>Npy)&(Ndx>Npx),   
    delta = aux(Ndy/2+1-Npy/2:Ndy/2+Npy/2,Ndx/2+1-Npx/2:Ndx/2+Npx/2);
elseif (Ndy<=Npy)&(Ndx>Npx),   
    delta(Npy/2+1-Ndy/2:Npy/2+1+Ndy/2-1,:) = aux(:,Ndx/2+1-Npx/2:Ndx/2+Npx/2);
elseif (Ndy>Npy)&(Ndx<=Npx),   
    delta(:,Npx/2+1-Ndx/2:Npx/2+1+Ndx/2-1) = aux(Ndy/2+1-Npy/2:Ndy/2+1+Npy/2-1,:);
end 

if repres1 == 'w',
    PS = abs(fft2(delta)).^2;
    PS = fftshift(PS);    
    % noise, to be used only with translation variant transforms (such as orthogonal wavelet)
    delta = real(ifft2(sqrt(PS).*exp(j*angle(fft2(randn(size(PS)))))));
end    

%Boundary handling: it extends im and delta
if boundary,
    im = bound_extension(im,By,Bx,'mirror');
    if repres1 == 'w',
        delta = bound_extension(delta,By,Bx,'mirror');
    else    
        aux = delta;
        delta = zeros(Npy + 2*By, Npx + 2*Bx);
        delta(By+1:By+Npy,Bx+1:Bx+Npx) = aux;
    end    
else
	By=0;Bx=0;
end

delta = delta/sqrt(mean2(delta.^2));    % Normalize the energy (the noise variance is given by "sig")
delta = sig*delta;                      % Impose the desired variance to the noise


% main
t1 = clock;
if repres1 == 's',  % standard steerable pyramid
    im_d = decomp_reconst(im, Nsc, Nor, [blSzX blSzY], delta, parent,covariance,optim,sig);
elseif repres1 == 'fs', % full steerable pyramid    
    im_d = decomp_reconst_full(im, Nsc, Nor, [blSzX blSzY], delta, parent, covariance, optim, sig);    
elseif repres1 == 'w',  % orthogonal wavelet    
    if ~exist('repres2'),
        repres2 = 'daub1';
    end    
    filter = repres2;
    im_d = decomp_reconst_W(im, Nsc, filter, [blSzX blSzY], delta, parent, covariance, optim, sig);
elseif repres1 == 'uw',    % undecimated daubechies wavelet
    if ~exist('repres2'),
        repres2 = 'daub1';
    end
    if repres2(1:4) == 'haar',
        daub_order = 2;
    else    
        daub_order = 2*str2num(repres2(5));
    end    
    im_d = decomp_reconst_WU(im, Nsc, daub_order, [blSzX blSzY], delta, parent, covariance, optim, sig);
else
    error('Invalid representation parameter. See help info.');    
end    
t2 = clock;
elaps = t2 - t1;
elaps(4)*3600+elaps(5)*60+elaps(6); % elapsed time, in seconds

im_d = im_d(By+1:By+Npy,Bx+1:Bx+Npx);   
im_d = im_d(1:Ny,1:Nx);
