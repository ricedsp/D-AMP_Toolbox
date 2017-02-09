function test_suite = testCos
initTestSuite;

function testTooManyInputs
assertExceptionThrown(@() cos(1, 2), 'MATLAB:maxrhs');