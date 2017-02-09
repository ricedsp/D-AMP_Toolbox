%% <../index.html MATLAB xUnit Test Framework>: How to Run a Specific Test
% To run all the test cases in just one M-file, ignoring other test
% cases that might be in other files in the same directory, give
% the name of the file (without the ".m" extension) as an argument
% to |runtests|.
%
% For example

cd example_subfunction_tests

runtests testFliplr

%%
% To run a single test case, add the name of the test case using a
% colon (":"), like this:

runtests testFliplr:testFliplrVector

%%
% <../index.html Back to MATLAB xUnit Test Framework>

%%
% Copyright 2008-2010 The MathWorks, Inc.
