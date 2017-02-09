function ts = shrink(t,f)

%	im_shr = shrink(im0, f)
%
% It shrinks (spatially) an image into a factor f
% in each dimension. It does it by cropping the
% Fourier transform of the image.

% JPM, 5/1/95.

% Revised so it can work also with exponents of 3 factors: JPM 5/2003


[my,mx] = size(t);
T = fftshift(fft2(t))/f^2;
Ts = zeros(my/f,mx/f);

cy = ceil(my/2);
cx = ceil(mx/2);
evenmy = (my/2==floor(my/2));
evenmx = (mx/2==floor(mx/2));

y1 = cy + 2*evenmy - floor(my/(2*f));
y2 = cy + floor(my/(2*f));
x1 = cx + 2*evenmx - floor(mx/(2*f));
x2 = cx + floor(mx/(2*f));

Ts(1+evenmy:my/f,1+evenmx:mx/f)=T(y1:y2,x1:x2);
if evenmy,
    Ts(1+evenmy:my/f,1)=(T(y1:y2,x1-1)+T(y1:y2,x2+1))/2;
end
if evenmx,
    Ts(1,1+evenmx:mx/f)=(T(y1-1,x1:x2)+T(y2+1,x1:x2))/2;
end
if evenmy & evenmx,
    Ts(1,1)=(T(y1-1,x1-1)+T(y1-1,x2+1)+T(y2+1,x1-1)+T(y2+1,x2+1))/4;
end    
Ts = fftshift(Ts);
Ts = shift(Ts, [1 1] - [evenmy evenmx]);
ts = ifft2(Ts);
