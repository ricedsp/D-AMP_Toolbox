function test_suite = testBadSinTest
initTestSuite;

function testSinPi
% Example of a failing test case.  The test writer should have used
% assertAlmostEqual here.
assertEqual(sin(pi), 0);
