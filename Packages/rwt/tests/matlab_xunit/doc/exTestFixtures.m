%% <../index.html MATLAB xUnit Test Framework>: How to Write Tests That Share Common Set-Up Code
% Sometimes you want to write a set of test cases in which the same
% set of initialization steps is performed before each test case, or
% in which the same set of cleanup steps is performed after each
% test case.  This set of common _setup_ and _teardown_ code is
% called a _test fixture_.
%
% In subfunction-based test files, you can add subfunctions whose
% names begin with "setup" and "teardown".  These functions will be
% called before and after every test-case subfunction is called.  If
% the setup function returns an output argument, that value is saved
% and passed to every test-case subfunction and also to the teardown
% function.
%
% This example shows a setup function that creates a figure and 
% returns its handle.  The figure handle is passed to each test-case
% subfunction.  The figure handle is also passed to the teardown
% function, which cleans up after each test case by deleting the
% figure.

cd examples_general
type testSetupExample

%%
% Run the tests using |runtests|.

runtests testSetupExample

%%
% You might also want to see the 
% <./exTestCase.html example on writing test cases by
% subclassing TestCase>.

%%
% <../index.html Back to MATLAB xUnit Test Framework>

%%
% Copyright 2008-2010 The MathWorks, Inc.