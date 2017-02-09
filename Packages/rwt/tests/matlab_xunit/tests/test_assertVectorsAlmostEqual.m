function suite = test_assertVectorsAlmostEqual
initTestSuite;

%===============================================================================
function test_happyCase

A = [1 1e10];
B = [2 1e10];
% All code here should execute with no error.
assertVectorsAlmostEqual(A, B);
assertVectorsAlmostEqual(A, B, 'custom message');

%===============================================================================
function test_failedAssertion

A = [1 1e6];
B = [2 1e6];

f = @() assertVectorsAlmostEqual(A, B);
assertExceptionThrown(f, 'assertVectorsAlmostEqual:tolExceeded');

%===============================================================================
function test_failedAssertionWithCustomMessage

A = [1 1e6];
B = [2 1e6];
f = @() assertVectorsAlmostEqual(A, B, 'my message');
assertExceptionThrown(f, 'assertVectorsAlmostEqual:tolExceeded');

%===============================================================================
function test_nonFloatInputs()
assertExceptionThrown(@() assertVectorsAlmostEqual('hello', 'world'), ...
    'assertVectorsAlmostEqual:notFloat');

%===============================================================================
function test_sizeMismatch()
assertExceptionThrown(@() assertVectorsAlmostEqual(1, [1 2]), ...
    'assertVectorsAlmostEqual:sizeMismatch');

%===============================================================================
function test_finiteAndInfinite()
assertExceptionThrown(@() assertVectorsAlmostEqual([1 2], [1 Inf]), ...
    'assertVectorsAlmostEqual:tolExceeded');

%===============================================================================
function test_infiniteAndInfinite
assertExceptionThrown(@() assertVectorsAlmostEqual([1 Inf], [1 Inf]), ...
    'assertVectorsAlmostEqual:tolExceeded');

%===============================================================================
function test_finiteAndNaN
assertExceptionThrown(@() assertVectorsAlmostEqual([1 2], [1 NaN]), ...
    'assertVectorsAlmostEqual:tolExceeded');

%===============================================================================
function test_NanAndNan
assertExceptionThrown(@() assertVectorsAlmostEqual([1 NaN], [1 NaN]), ...
    'assertVectorsAlmostEqual:tolExceeded');

%===============================================================================
function test_oppositeSignInfs
assertExceptionThrown(@() assertVectorsAlmostEqual([1 Inf], [1 -Inf]), ...
    'assertVectorsAlmostEqual:tolExceeded');


