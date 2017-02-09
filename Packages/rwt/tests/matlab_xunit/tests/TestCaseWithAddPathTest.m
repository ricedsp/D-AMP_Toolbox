%TestCaseTest Unit tests for the TestCaseWithAddPath class

%   Steven L. Eddins
%   Copyright The MathWorks 2008

classdef TestCaseWithAddPathTest < TestCaseWithAddPath

    methods
        function self = TestCaseWithAddPathTest(name)
            self = self@TestCaseWithAddPath(name, ...
                fullfile(fileparts(which(mfilename)), 'helper_classes'));
        end

        function testPath(self)
            % Verify that a function in helper_classes is seen on the path.
            assertEqual(exist('testFunctionHandlesA', 'file'), 2);
        end
        
        function testRunTestOnPath(self)
            % Verify that we can make a test suite and run it using a file
            % in the new path directory.
            logger = TestRunLogger();
            suite = TestSuite('testFunctionHandlesA');
            did_pass = suite.run(logger);
            assertTrue(did_pass);
        end
    end

end