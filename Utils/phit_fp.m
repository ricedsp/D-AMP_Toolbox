function x_hat=phit_fp(y,m,n,ops)
% function x_hat=phit_fp(y,m,n,ops)
% PHIT_FP projects a length m signal onto the transpose of an mxn measurment matrix.
% Input:
%       y   : length m signals (for example a vectorized image)
%       m   : the number of measurements
%       n   : the signal length
%       ops : currently unused argument
%Output:
%       x   : result of projecting y onto phi'.
    x_hat=zeros(n,1);
    remaining_rows=n;
    K=4096;
    iters=ceil(n/K);
    for i=0:iters-1
        rng(i);
        row_num=min(K,remaining_rows);
        phi_columns=randn(m,row_num); 
        column_norms=sqrt(sum(abs(phi_columns).^2,1));
        phi_columns=bsxfun(@rdivide,phi_columns,column_norms);
        phit_rows=phi_columns';
        x_hat(K*i+1:K*i+row_num)=phit_rows*y;
        remaining_rows=remaining_rows-K;
    end
end