%% <../index.html MATLAB xUnit Test Framework>: How to Run Tests in a Package
% To run all the test cases in a package, give the name of the
% package as an argument to |runtests|. *Note:* Running tests in a package
% requires MATLAB R2009a or later.
%
% For example, suppose you are distributing a set of MATLAB files called the
% "ABC Toolbox." Then you could put your tests inside a package called abc_tests
% and run them like this:

runtests abc_tests

%%
% (Note that the initial "+" character in the name of the package folder on disk
% is not part of the package name.)
%
% Or you could put your tests inside a subpackage called abc.tests and run them
% like this:

runtests abc.tests

%%
% You should not use a generic top-level package name such "tests" because then
% your package might be unintentionally combined with packages with the same
% name created by other people.  

%%
% <../index.html Back to MATLAB xUnit Test Framework>

%%
% Copyright 2010 The MathWorks, Inc.
