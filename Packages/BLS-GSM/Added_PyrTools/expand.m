function te = expand(t,f)

%	im_exp = expand(im0, f)
%
% It expands (spatially) an image into a factor f
% in each dimension. It does it filling in with zeros
% the expanded Fourier domain.

% JPM, 5/1/95.

% Revised so it can work also with exponents of 3 factors: JPM 5/2003

[my mx] = size(t);
my = f*my;
mx = f*mx;
Te = zeros(my,mx);
T = f^2*fftshift(fft2(t));

cy = ceil(my/2);
cx = ceil(mx/2);
evenmy = (my/2==floor(my/2));
evenmx = (mx/2==floor(mx/2));

y1 = cy + 2*evenmy - floor(my/(2*f));
y2 = cy + floor(my/(2*f));
x1 = cx + 2*evenmx - floor(mx/(2*f));
x2 = cx + floor(mx/(2*f));

Te(y1:y2,x1:x2)=T(1+evenmy:my/f,1+evenmx:mx/f);
if evenmy,
    Te(y1-1,x1:x2)=T(1,2:mx/f)/2;
    Te(y2+1,x1:x2)=((T(1,mx/f:-1:2)/2)').';
end
if evenmx,
    Te(y1:y2,x1-1)=T(2:my/f,1)/2;
    Te(y1:y2,x2+1)=((T(my/f:-1:2,1)/2)').';
end
if evenmx & evenmy,
    esq=T(1,1)/4;
    Te(y1-1,x1-1)=esq;
    Te(y1-1,x2+1)=esq;
    Te(y2+1,x1-1)=esq;
    Te(y2+1,x2+1)=esq;
end    
Te=fftshift(Te);
Te = shift(Te, [1 1] - [evenmy evenmx]);
te=ifft2(Te);
