%% <../index.html MATLAB xUnit Test Framework>: How to Run Tests Silently and Query the Results
% When you run a test suite using |runtests|, the results are
% summarized in the Command Window.  This example shows you how to
% run a test suite so that nothing prints to the Command Window, and
% it shows you how to write a program to automatically determine the
% results of running the test suite.
%
% There are four steps to follow.
%
% 1. Construct a |TestSuite| object.  In this example we'll use the |fromPwd|
% method of the |TestSuite| class to construct a test suite using all the test
% cases found in the |examples_general| directory.

cd examples_general
suite = TestSuite.fromPwd();

%%
% You can look up information about the individual test cases.

suite.TestComponents{1}

%%
% You can see above that the first test component in the test suite is itself
% another test suite, which contains the test cases defined by the M-file named
% TestUsingTestCase. Here's what one of these individual test cases looks like:

suite.TestComponents{1}.TestComponents{1}

%%
% 2. Construct a TestLogger object.  This object can receive
% notifications about what happens when a test suite is executed.

logger = TestRunLogger;

%%
% 3. Call the |run| method of the |TestSuite| object, passing it the
% logger.

suite.run(logger);

%%
% The |TestLogger| object can now be queried to determine what
% happened during the test.

logger

%%
% There were eight test cases run (logger.NumTestCases), resulting in
% one test failure and one test error.  Detailed information about
% what went wrong can be found in |logger.Faults|.

logger.Faults(1)

%%

logger.Faults(2)

%%
% You can drill further to determine the names of the failing tests,
% as well as the complete stack trace associated with each failure.

logger.Faults(1).TestCase

%%

logger.Faults(1).Exception.stack(1)

%%

logger.Faults(1).Exception.stack(2)

%%
% <../index.html Back to MATLAB xUnit Test Framework>

%%
% Copyright 2008-2010 The MathWorks, Inc.
