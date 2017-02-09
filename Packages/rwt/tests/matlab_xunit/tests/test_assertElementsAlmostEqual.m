function suite = test_assertElementsAlmostEqual
initTestSuite;

%===============================================================================
function test_happyCase

% All code here should execute with no error.
assertElementsAlmostEqual(1, 1 + sqrt(eps)/10);
assertElementsAlmostEqual(1, 1 + sqrt(eps)/10, 'custom message');

%===============================================================================
function test_failedAssertion

f = @() assertElementsAlmostEqual(1, 1 + 10*sqrt(eps));
assertExceptionThrown(f, 'assertElementsAlmostEqual:tolExceeded');

%===============================================================================
function test_nonFloatInputs()
assertExceptionThrown(@() assertElementsAlmostEqual('hello', 'world'), ...
    'assertElementsAlmostEqual:notFloat');

%===============================================================================
function test_sizeMismatch()
assertExceptionThrown(@() assertElementsAlmostEqual(1, [1 2]), ...
    'assertElementsAlmostEqual:sizeMismatch');

function test_finiteAndInfinite()
assertExceptionThrown(@() assertElementsAlmostEqual(1, Inf), ...
    'assertElementsAlmostEqual:tolExceeded');

function test_infiniteAndInfinite()
assertElementsAlmostEqual(Inf, Inf);

function test_finiteAndNaN()
assertExceptionThrown(@() assertElementsAlmostEqual(1, NaN), ...
    'assertElementsAlmostEqual:tolExceeded');

function test_nanAndNaN()
assertElementsAlmostEqual(NaN, NaN);

function test_plusMinusInfinity()
assertExceptionThrown(@() assertElementsAlmostEqual(+Inf, -Inf), ...
    'assertElementsAlmostEqual:tolExceeded');

function test_infiniteAndNaN()
assertExceptionThrown(@() assertElementsAlmostEqual(Inf, NaN), ...
    'assertElementsAlmostEqual:tolExceeded');



