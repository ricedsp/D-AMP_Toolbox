function test_suite = test_TestSuiteInDir
%test_TestSuiteInDir Unit test for TestSuiteInDir class.

%   Steven L. Eddins
%   Copyright 2009 The MathWorks, Inc.

initTestSuite;

function test_constructor
this_test_path = fileparts(which(mfilename));
cwd_test_dir = fullfile(this_test_path, 'cwd_test');
suite = TestSuiteInDir(cwd_test_dir);

assertEqual(suite.Name, 'cwd_test');
assertEqual(suite.Location, cwd_test_dir);

function test_gatherTestCases
this_test_path = fileparts(which(mfilename));
cwd_test_dir = fullfile(this_test_path, 'cwd_test');
suite = TestSuiteInDir(cwd_test_dir);
suite.gatherTestCases();

assertEqual(numel(suite.TestComponents), 3);

