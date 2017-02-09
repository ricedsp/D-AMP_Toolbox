function result = compareFloats(varargin)
%compareFloats Compare floating-point arrays using tolerance.
%   result = compareFloats(A, B, compare_type, tol_type, tol, floor_tol)
%   compares the floating-point arrays A and B using a tolerance.  compare_type
%   is either 'elementwise' or 'vector'.  tol_type is either 'relative' or
%   'absolute'.  tol and floor_tol are the scalar tolerance values.
%
%   There are four different tolerance tests used, depending on the comparison
%   type and the tolerance type:
%
%   1. Comparison type: 'elementwise'     Tolerance type: 'relative'
%
%       all( abs(A(:) - B(:)) <= tol * max(abs(A(:)), abs(B(:))) + floor_tol )
%
%   2. Comparison type: 'elementwise'     Tolerance type: 'absolute'
%
%       all( abs(A(:) - B(:) <= tol )
%
%   3. Comparison type: 'vector'          Tolerance type: 'relative'
%
%       norm(A(:) - B(:) <= tol * max(norm(A(:)), norm(B(:))) + floor_tol
%
%   4. Comparison type: 'vector'          Tolerance type: 'absolute'
%
%       norm(A(:) - B(:)) <= tol
%
%   Note that floor_tol is not used when the tolerance type is 'absolute'.
%
%   compare_type, tol_type, tol, and floor_tol are all optional inputs.  The
%   default value for compare_type is 'elementwise'.  The default value for
%   tol_type is 'relative'.  If both A and B are double, then the default value
%   for tol is sqrt(eps), and the default value for floor_tol is eps.  If either
%   A or B is single, then the default value for tol is sqrt(eps('single')), and
%   the default value for floor_tol is eps('single').
%
%   If A or B is complex, then the tolerance test is applied independently to
%   the real and imaginary parts.

%   Steven L. Eddins
%   Copyright 2008-2009 The MathWorks, Inc.

if nargin >= 3
    % compare_type specified.  Grab it and then use parseFloatAssertInputs to
    % process the remaining input arguments.
    compare_type = varargin{3};
    varargin(3) = [];
    if isempty(strcmp(compare_type, {'elementwise', 'vector'}))
        error('MTEST:compareFloats:unrecognizedCompareType', ...
            'COMPARE_TYPE must be ''elementwise'' or ''vector''.');
    end
else
    compare_type = 'elementwise';
end

params = mtest.utils.parseFloatAssertInputs(varargin{:});

A = params.A(:);
B = params.B(:);

[A, B] = preprocessNanInf(A, B);

switch compare_type
    case 'elementwise'
        magFcn = @abs;
        
    case 'vector'
        magFcn = @norm;
        
    otherwise
        error('MTEST:compareFloats:unrecognizedCompareType', ...
            'COMPARE_TYPE must be ''elementwise'' or ''vector''.');
end

switch params.ToleranceType
    case 'relative'
        compareFcn = @(A, B) magFcn(A - B) <= ...
            params.Tolerance * max(magFcn(A), magFcn(B)) + ...
            params.FloorTolerance;
        
    case 'absolute'
        compareFcn = @(A, B) magFcn(A - B) <= params.Tolerance;
        
    otherwise
        error('MTEST:compareFloats:unrecognizedToleranceType', ...
            'TOL_TYPE must be ''relative'' or ''absolute''.');
end

if isreal(A) && isreal(B)
    result = compareFcn(A, B);
else
    result = compareFcn(real(A), real(B)) & compareFcn(imag(A), imag(B));
end

result = all(result);

%===============================================================================
function [A, B] = preprocessNanInf(A, B)

make_zero = isnan(A) & isnan(B);
make_zero = make_zero | ((A == Inf) & (B == Inf));
make_zero = make_zero | ((A == -Inf) & (B == -Inf));

A(make_zero) = 0;
B(make_zero) = 0;
