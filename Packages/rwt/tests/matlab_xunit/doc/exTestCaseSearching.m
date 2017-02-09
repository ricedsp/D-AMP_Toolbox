%% <../index.html MATLAB xUnit Test Framework>: How RUNTESTS Searches for Test Cases
% When you call |runtests| with no input arguments:
%
%   >> runtests
%
% it automatically searches for all the test cases in the current directory.  It
% looks for test cases in three types of M-files:
%
% 1. An M-file function whose name begins or ends with "test" or "Test" and that does
% not return an output argument.  Such a function is considered to be a single
% test case. 
%
% 2. An M-file function whose name begins or ends with "test" or "Test" and that returns
% an output argument that is a test suite.  Such a function is considered to contain
% subfunction-style test cases.  Each subfunction whose name begins or ends with "test"
% or "Test" is a test case. 
%
% 3. An M-file that defines a subclass of TestCase.  Each method beginning or ending with
% "test" or "Test" is a test case.
%
% |runtests| uses the |TestSuite| static methods |fromName| and |fromPwd| to
% automatically construct the test suites.
%
% Here are a couple of examples.
%
% |TestSuite.fromName| takes an M-file name, determines what
% kind of test file it is, and returns a cell array of test case objects.

cd examples_general
test_suite_1 = TestSuite.fromName('testSetupExample')

%%
% |TestSuite.fromPwd| returns a test suite based on all the test files in the
% current directory.

test_suite_2 = TestSuite.fromPwd()

%%
% <../index.html Back to MATLAB xUnit Test Framework>

%%
% Copyright 2008-2010 The MathWorks, Inc.