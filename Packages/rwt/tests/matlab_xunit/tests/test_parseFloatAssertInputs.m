function suite = test_parseFloatAssertInputs
initTestSuite;

%===============================================================================
function test_tooFewInputs()
assertExceptionThrown(@() xunit.utils.parseFloatAssertInputs(), ...
    'MATLAB:nargchk:notEnoughInputs');

%===============================================================================
function test_tooManyInputs()
assertExceptionThrown(@() xunit.utils.parseFloatAssertInputs(1,2,3,4,5,6,7), ...
    'MATLAB:nargchk:tooManyInputs');

%===============================================================================
function test_twoInputs()
params = xunit.utils.parseFloatAssertInputs(1, 2);
assertEqual(params.A, 1);
assertEqual(params.B, 2);
assertEqual(params.ToleranceType, 'relative');
assertEqual(params.Tolerance, sqrt(eps));
assertEqual(params.FloorTolerance, sqrt(eps));
assertEqual(params.Message, '');

%===============================================================================
function test_threeInputs()
expected.A = 1;
expected.B = 2;
expected.ToleranceType = 'relative';
expected.Tolerance = sqrt(eps);
expected.FloorTolerance = sqrt(eps);
expected.Message = '';

params = xunit.utils.parseFloatAssertInputs(1, 2, 'relative');
assertEqual(params, expected);

params = xunit.utils.parseFloatAssertInputs(1, 2, 'absolute');
expected.ToleranceType = 'absolute';
assertEqual(params, expected);

params = xunit.utils.parseFloatAssertInputs(1, 2, 'message');
expected.ToleranceType = 'relative';
expected.Message = 'message';
assertEqual(params, expected);

%===============================================================================
function test_fourInputs()
expected.A = 1;
expected.B = 2;
expected.ToleranceType = 'absolute';
expected.Tolerance = sqrt(eps);
expected.FloorTolerance = sqrt(eps);
expected.Message = '';

params = xunit.utils.parseFloatAssertInputs(1, 2, 'absolute', 0.1);
expected.Tolerance = 0.1;
assertEqual(params, expected);

params = xunit.utils.parseFloatAssertInputs(1, 2, 'absolute', 'message');
expected.Tolerance = sqrt(eps);
expected.Message = 'message';
assertEqual(params, expected);

%===============================================================================
function test_fiveInputs()
expected.A = 1;
expected.B = 2;
expected.ToleranceType = 'absolute';
expected.Tolerance = 0.1;
expected.FloorTolerance = 0.05;
expected.Message = '';

params = xunit.utils.parseFloatAssertInputs(1, 2, 'absolute', 0.1, 0.05);
assertEqual(params, expected);

params = xunit.utils.parseFloatAssertInputs(1, 2, 'absolute', 0.1, 'message');
expected.FloorTolerance = sqrt(eps);
expected.Message = 'message';
assertEqual(params, expected);

%===============================================================================
function test_sixInputs()
expected.A = 1;
expected.B = 2;
expected.ToleranceType = 'absolute';
expected.Tolerance = 0.1;
expected.FloorTolerance = 0.05;
expected.Message = 'message';

params = xunit.utils.parseFloatAssertInputs(1, 2, 'absolute', 0.1, 0.05, 'message');
assertEqual(params, expected);

%===============================================================================
function test_twoSingleInputs()
expected.A = 1;
expected.B = 2;
expected.ToleranceType = 'relative';
expected.Tolerance = sqrt(eps('single'));
expected.FloorTolerance = sqrt(eps('single'));
expected.Message = '';

params = xunit.utils.parseFloatAssertInputs(single(1), single(2));
assertEqual(params, expected);

%===============================================================================
function test_twoSingleAndDoubleInputs()
expected.A = 1;
expected.B = 2;
expected.ToleranceType = 'relative';
expected.Tolerance = sqrt(eps('single'));
expected.FloorTolerance = sqrt(eps('single'));
expected.Message = '';

params = xunit.utils.parseFloatAssertInputs(single(1), double(2));
assertEqual(params, expected);


