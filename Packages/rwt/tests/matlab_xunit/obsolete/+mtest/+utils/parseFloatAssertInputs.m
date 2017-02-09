function params = parseFloatAssertInputs(varargin)
%parseFloatAssertInputs Parse inputs for floating-point assertion functions.
%   params = parseFloatAssertInputs(varargin) parses the input arguments for
%   assertElementsAlmostEqual, assertVectorsAlmostEqual, and compareFcn. It
%   returns a parameter struct containing the fields:
%
%       A    B    ToleranceType    Tolerance    FloorTolerance

%   Steven L. Eddins
%   Copyright 2008-2009 The MathWorks, Inc.

error(nargchk(2, 6, nargin, 'struct'));

params = struct('A', {[]}, 'B', {[]}, 'ToleranceType', {[]}, ...
    'Tolerance', {[]}, 'FloorTolerance', {[]}, 'Message', {''});

% The first two input arguments are always A and B.
params.A = varargin{1};
params.B = varargin{2};
varargin(1:2) = [];

% If the last argument is a message string, process it and remove it from the list.
if (numel(varargin) >= 1) && ischar(varargin{end}) && ...
        ~any(strcmp(varargin{end}, {'relative', 'absolute'}))
    params.Message = varargin{end};
    varargin(end) = [];
end

checkAB(params.A, params.B);

epsilon = max(eps(class(params.A)), eps(class(params.B)));

if numel(varargin) < 3
    % floor_tol not specified; set default.
    params.FloorTolerance = epsilon;
else
    params.FloorTolerance = varargin{3};
end

if numel(varargin) < 2
    % tol not specified; set default.
    params.Tolerance = sqrt(epsilon);
else
    params.Tolerance = varargin{2};
end

if numel(varargin) < 1
    % tol_type not specified; set default.
    params.ToleranceType = 'relative';
else
    params.ToleranceType = varargin{1};
end

%===============================================================================
function checkAB(A, B)
if ~isfloat(A) || ~isfloat(B)
    error('MTEST:parseFloatAssertInputs:inputsNotFloat', ...
        'A and B must be floating-point arrays.');
end

if ~isequal(size(A), size(B))
    error('MTEST:parseFloatAssertInputs:sizeMismatch', ...
        'A and B must have the same size.');
end