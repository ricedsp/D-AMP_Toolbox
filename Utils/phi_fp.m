function [y]=phi_fp(x,m,n,ops)
% function [y]=phi_fp(x,m,n,ops)
% PHI_FP project length n signal x onto an mxn random gaussian matrix 
% Input:
%       x   : length n signals (for example a vectorized image)
%       m   : the number of measurements
%       n   : the signal length
%       ops : currently unused argument
%Output:
%       y   : result of projecting x onto phi.
    y=zeros(m,1);
    remaining_columns=n;
    K=4096;%How large each submatrtix is.  Determines memory usage
    iters=ceil(n/K);
    for i=0:iters-1
        rng(i);
        col_num=min(K,remaining_columns);
        phi_columns=randn(m,col_num);
        column_norms=sqrt(sum(abs(phi_columns).^2,1));
        phi_columns=bsxfun(@rdivide,phi_columns,column_norms);
        y=y+phi_columns*x(K*i+1:K*i+col_num);
        remaining_columns=remaining_columns-K;
    end
end