%% <../index.html MATLAB xUnit Test Framework>: How to Put Multiple Test Cases in One M-file
% The Quick Start example showed how you can write a simple M-file
% to be a single test case.  This example shows you how to put multiple
% test cases in one M-file.
%
% Name your M-file beginning or ending with "test", like
% "testMyFunc".  Start by putting the following two lines at the
% beginning of the file.  It's important that the output variable
% name on line 1 be |test_suite|.
%
%    function test_suite = testMyFunc
%    initTestSuite;
%
% Next, add subfunctions to the file.  Each subfunction beginning
% or ending with "test" becomes an individual test case.
%
% The directory example_subfunction_tests contains a test M-file
% containing subfunction test cases for the |fliplr| function.

cd example_subfunction_tests

type testFliplr

%%
% As usual, run the test cases using |runtests|:

runtests

%%
% <../index.html Back to MATLAB xUnit Test Framework>

%%
% Copyright 2008-2010 The MathWorks, Inc.