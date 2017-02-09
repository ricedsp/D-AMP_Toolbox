function [ X ] = psi_fp( x,h,n,image)
% function [ X ] = psi_fp( x,h,n,image)
% PSI_FP maps coefficients to pixels
% Input:
%       x       : coefficient values
%       h   : wavelet transform filter coefficients
%       n   : signal size
%       image  : 1 or 0 based on whether or not x is a square image
%Output:
%       X   : the pixel values associated with the wavelet coefficients x
if image==1
    x=reshape(x,[sqrt(n),sqrt(n)]);
end
X=midwt(x,h,log2(sqrt(n)));
X=X(:);
end

