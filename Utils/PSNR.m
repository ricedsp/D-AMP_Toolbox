function [psnr]=PSNR(x,x_hat)
% function [psnr]=PSNR(x,x_hat)
% PSNR computes the psnr of x_hat; an estimate of x.
% Input:
%       x     : the true signal
%       x_hat : an estimate of x
%Output:
%       psnr  : the PSNR of x_hat
    [imheight, imwidth]=size(x);
    MSE=sum(sum((double(x)-double(x_hat)).^2))/(imheight*imwidth);
    psnr=10*log(255^2/MSE)/log(10);