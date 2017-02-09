%TestCaseTest Unit tests for the TestCase class

%   Steven L. Eddins
%   Copyright The MathWorks 2008

classdef TestCaseTest < TestCaseInDir

    methods
        function self = TestCaseTest(name)
            self = self@TestCaseInDir(name, ...
                fullfile(fileparts(which(mfilename)), 'helper_classes'));
        end

        function testConstructor(self)
            % Exercise the constructor.  Verify that the Name and Location
            % properties are set correctly.
            tc = TwoPassingTests('testMethod1');
            assertEqual(tc.Name, 'testMethod1');
            assertEqual(tc.Location, which('TwoPassingTests'));
        end

        function testPassingTests(self)
            % Verify that the expected observer notifications are received in
            % the proper order.
            logger = TestRunLogger();
            TestSuite('TwoPassingTests').run(logger);
            assertTrue(isequal(logger.Log, ...
                {'TestRunStarted', 'TestComponentStarted', ...
                'TestComponentStarted', 'TestComponentFinished', ...
                'TestComponentStarted', 'TestComponentFinished', ...
                'TestComponentFinished', 'TestRunFinished'}));
        end

        function testFixtureCalls(self)
            % Verify that fixture calls are made in the proper order.
            tc = LoggingTestCase('testMethod');
            tc.run(TestRunLogger());
            assertTrue(isequal(tc.log, {'setUp', 'testMethod', 'tearDown'}));
        end

        function testTestFailure(self)
            % Verify that a test failure is recorded.
            logger = TestRunLogger();
            TestSuite('FailingTestCase').run(logger);
            assertTrue(isequal(logger.NumFailures, 1));
        end

        function testTestError(self)
            % Verify that a test error is recorded.
            logger = TestRunLogger();
            TestSuite('BadFixture').run(logger);
            assertTrue(isequal(logger.NumErrors, 1));
        end

    end

end