function suite = test_compareFloats
initTestSuite;

%===============================================================================
function test_elementwiseRelativeTolerance

tol = 0.1;
floor_tol = 0.01;

assertTrue(xunit.utils.compareFloats([10 20], [11 20], 'elementwise', ...
    'relative', tol, floor_tol));
assertFalse(xunit.utils.compareFloats([10 20], [11.2 20], 'elementwise', ...
    'relative', tol, floor_tol));

% Verify floor tolerance
assertTrue(xunit.utils.compareFloats([0.001 1], [0.010 1], 'elementwise', ...
    'relative', tol, floor_tol));

%===============================================================================
function test_elementwiseAbsoluteTolerance

assertTrue(xunit.utils.compareFloats([10 20], [10.1 20], 'elementwise', ...
    'absolute', 0.1));
assertFalse(xunit.utils.compareFloats([10 20], [10.1001 20], 'elementwise', ...
    'absolute', 0.1));

%===============================================================================
function test_vectorRelativeTolerance

% The A-B pair below would fail an elementwise test.
A = [1 10];
B = [1.5 10];
tol = 0.05;

assertTrue(xunit.utils.compareFloats(A, B, 'vector', 'relative', tol));

B = [1.6 10];
assertFalse(xunit.utils.compareFloats(A, B, 'vector', 'relative', tol));

%===============================================================================
function test_vectorAbsoluteTolerance

A = [1 10];
B = [1.4 10];

assertTrue(xunit.utils.compareFloats(A, B, 'vector', 'absolute', 0.5));
assertFalse(xunit.utils.compareFloats(A, B, 'vector', 'absolute', 0.3));

%===============================================================================
function test_NaNs

% NaNs in the same spots are OK.
A = [1 1 1 NaN 1 1 1 NaN 1];
B = [1 1 1 NaN 1 1 1 NaN 1];

assertTrue(xunit.utils.compareFloats(A, B));

% NaNs in different spots are not OK.
B2 = [1 1 NaN NaN 1 1 1 NaN 1];
assertFalse(xunit.utils.compareFloats(A, B2));

%===============================================================================
function test_Infs

% Infinities in the same locations are OK if they have the same sign.
assertTrue(xunit.utils.compareFloats([1 2 3 Inf 4 5], [1 2 3 Inf 4 5]));
assertTrue(xunit.utils.compareFloats([1 2 3 -Inf 4 5], [1 2 3 -Inf 4 5]));
assertFalse(xunit.utils.compareFloats([1 2 3 Inf 4 5], [1 2 3 -Inf 4 5], ...
    'elementwise', 'absolute'));

%===============================================================================
function test_complexInput

% Real and imaginary parts are compared separately.
assertTrue(xunit.utils.compareFloats(1, 1+0.09i, 'elementwise', 'absolute', 0.1));
assertFalse(xunit.utils.compareFloats(1, 1+0.11i, 'elementwise', 'absolute', 0.1));

%===============================================================================
function test_comparisonTypeSpecified

% Verify handling of third input argument, the comparison type.  The rest of the
% input syntax is handled by parseFloatAssertInputs and tested by the unit test
% for that function.

% The A-B pair below fails using elementwise comparison but passes using vector
% comparison.
A = [1.5 10];
B = [1 10];
tol = 0.1;

assertFalse(xunit.utils.compareFloats(A, B, 'elementwise', 'relative', tol));
assertTrue(xunit.utils.compareFloats(A, B, 'vector', 'relative', tol));
