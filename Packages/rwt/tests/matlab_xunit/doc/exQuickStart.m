%% <../index.html MATLAB xUnit Test Framework>: How to Write and Run Tests 
% This example shows how to write and run a couple of test cases for the MATLAB
% |fliplr| function.

%% Make a folder for your tests
% To get started, create a folder (directory) that will contain your tests, and
% then make that your working folder.  The test directory in this example is
% example_quick_start.

cd example_quick_start

%% Write each test case as a simple M-file
% Write each test case as an M-file function that returns no output arguments.
% The function name should start or end with "test" or "Test".  The test case
% passes if the function runs with no error.
%
% Here's a test-case M-file that verifies the correct output for a vector input.

type testFliplrVector

%%
% The function |testFliplrVector| calls the function being tested and checks the
% output against the expected output.  If the output is different than expected,
% the function calls |error|.
%
% Here's another test-case M-file that verifies the correct |fliplr| output for
% a matrix input.

type testFliplrMatrix

%%
% This function is simpler than |testFliplrVector| because it uses the utility
% testing function |assertEqual|.  |assertEqual| checks to see whether its two
% inputs are equal. If they are equal, |assertEqual| simply returns silently.
% If they are not equal, |assertEqual| calls |error|.

%% Run all the tests using |runtests|
% To run all your test cases, simply call |runtests|.  |runtests| automatically finds
% all the test cases in the current directory, runs them, and reports the
% results to the Command Window.

runtests

%%
% <../index.html Back to MATLAB xUnit Test Framework>

%%
% Copyright 2008-2010 The MathWorks, Inc.