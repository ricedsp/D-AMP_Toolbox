%TestFuncHandleTests TeseCase class used to test function-handle-based tests

%   Steven L. Eddins
%   Copyright 2008 The MathWorks, Inc.

classdef TestFuncHandleTests < TestCaseInDir

    methods
        function self = TestFuncHandleTests(name)
            self = self@TestCaseInDir(name, ...
                fullfile(fileparts(which(mfilename)), 'helper_classes'));
        end
        
        function testSuiteNameAndLocation(self)
            test_suite = testFunctionHandlesA();
            assertEqual(test_suite.Name, 'testFunctionHandlesA');
            assertEqual(test_suite.Location, which('testFunctionHandlesA'));
        end

        function testOutputs(self)
            % Exercise the function-handle test M-file. Output should be a
            % two-element cell array of TestCase objects.
            test_suite = testFunctionHandlesA();
            assertTrue(isa(test_suite, 'TestSuite'));
            assertEqual(test_suite.numTestCases(), 2);
        end

        function testCaseNames(self)
            % Verify that Name property of test cases is set properly.
            test_suite = testFunctionHandlesA();
            assertEqual(test_suite.TestComponents{1}.Name, 'testA');
            assertEqual(test_suite.TestComponents{2}.Name, 'testB');
        end

        function testCaseLocation(self)
            % Verify that the Location field of test cases is set properly.
            test_suite = testFunctionHandlesA();
            expected_location = which('testFunctionHandlesA');
            assertEqual(test_suite.TestComponents{1}.Location, expected_location);
            assertEqual(test_suite.TestComponents{2}.Location, expected_location);
        end

        function testPassingTests(self)
            % Verify that the expected observer notifications are received in
            % the proper order.
            logger = TestRunLogger();
            suite = testFunctionHandlesA;
            suite.run(logger);
            assertEqual(logger.Log, ...
                {'TestRunStarted', 'TestComponentStarted', ...
                'TestComponentStarted', 'TestComponentFinished', ...
                'TestComponentStarted', 'TestComponentFinished', ...
                'TestComponentFinished', 'TestRunFinished'});
        end

        function testTestFixture(self)
            % Verify that test fixture functions that use testData run without
            % error.  (See test assertions in testFunctionHandlesB.)
            logger = TestRunLogger();
            suite = testFunctionHandlesB;
            suite.run(logger);
            assertEqual(logger.NumFailures, 0);
            assertEqual(logger.NumErrors, 0);
        end

        function testTestFixtureError(self)
            % Verify that an exception thrown in a test fixture is recorded as a
            % test error.
            logger = TestRunLogger();
            suite = testFunctionHandlesC();
            suite.run(logger);
            assertEqual(logger.NumErrors, 2);
        end

        function testFixtureNoTestData(self)
            % Verify that when setupFcn returns no output argument, the test
            % functions and the teardown function are called with no inputs.
            % (See test assertions in testFunctionHandlesD.)
            logger = TestRunLogger();
            suite = testFunctionHandlesD();
            suite.run(logger);
            assertEqual(logger.NumFailures, 0);
            assertEqual(logger.NumErrors, 0);
        end
        
        function testFailingTest(self)
            % Verify that the expected observer notifications are received in
            % the proper order for a failing test.
            logger = TestRunLogger();
            suite = testFunctionHandlesE();
            suite.run(logger);
            assertEqual(logger.Log, ...
                {'TestRunStarted', 'TestComponentStarted', ...
                'TestComponentStarted', 'TestCaseFailure', 'TestComponentFinished', ...
                'TestComponentFinished', 'TestRunFinished'});
        end
        
        function testTeardownFcnButNoSetupFcn(self)
            % Verify that a test file works if it has a teardown function but no
            % setup function.
            logger = TestRunLogger();
            suite = testFunctionHandlesTeardownNoSetup();
            suite.run(logger);
            
            assertEqual(logger.NumTestCases, 1);
            assertEqual(logger.NumFailures, 0);
            assertEqual(logger.NumErrors, 0);
        end

    end
end