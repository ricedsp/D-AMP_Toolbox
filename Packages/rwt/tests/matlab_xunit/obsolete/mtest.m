function out = mtest(name)
%mtest Run unit tests
%   mtest runs all the test cases that can be found in the current directory and
%   summarizes the results in the Command Window.
%
%   Test cases can be found in the following places in the current directory:
%
%       * An M-file function whose name starts with "test" or "Test" that
%       returns no output arguments.
%
%       * An M-file function whose name starts with "test" or "Test" that
%       contains subfunction tests and uses the initTestSuite script to
%       return a TestSuite object.
%
%       * An M-file defining a subclass of TestCase.
%
%   mtest(mfilename) runs test cases found in the specified function or class
%   name. The function or class needs to be in the current directory or on the
%   MATLAB path.
%
%   mtest('mfilename:testname') runs the specific test case named 'testname'
%   found in the function or class 'name'.
%
%   mtest(dirname) runs all the test cases that can be found in the specified
%   directory.
%
%   Examples
%   --------
%   Find and run all the test cases in the current directory.
%
%       mtest
%
%   Find and run all the test cases contained in the M-file myfunc.
%
%       mtest myfunc
%
%   Find and run all the test cases contained in the TestCase subclass
%   MyTestCase.
%
%       mtest MyTestCase
%
%   Run the test case named 'testFeature' contained in the M-file myfunc.
%
%       mtest myfunc:testFeature
%
%   Run all the tests in a specific directory.
%
%       mtest c:\Work\MyProject\tests

%   Steven L. Eddins
%   Copyright 2008-2009 The MathWorks, Inc.

if nargin < 1
    suite = TestSuite.fromPwd();
else
    suite = TestSuite.fromName(name);
    
    user_gave_a_directory_name = isempty(suite.TestComponents) && ...
        (exist(name, 'file') == 7);
    if user_gave_a_directory_name
        % Before changing directories, arrange to restore the current directory
        % safely.
        currentDir = pwd;
        c = onCleanup(@() cd(currentDir));
        
        cd(name);
        suite = TestSuite.fromPwd();
    end
end

did_pass = suite.run(CommandWindowTestRunDisplay());

if nargout > 0
    out = did_pass;
end
