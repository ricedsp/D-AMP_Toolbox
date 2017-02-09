function [ x ] = psit_fp( X,h,n,image)
% function [ x ] = psit_fp( X,h,n,image)
% PSI_FP maps pixels to coefficients
% Input:
%       X       : pixel values
%       h   : wavelet transform filter coefficients
%       n   : signal size
%       image  : 1 or 0 based on whether or not x is a square image
%Output:
%       X   : the coefficient values associated with the pixel values X
if image==1
    X=reshape(X,[sqrt(n),sqrt(n)]);
end
x=mdwt(X,h,log2(sqrt(n)));
x=x(:);
end

