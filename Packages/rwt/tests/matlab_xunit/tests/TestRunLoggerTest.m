%TestSuiteTest Unit tests for TestSuite class

classdef TestRunLoggerTest < TestCaseInDir

   methods
      function self = TestRunLoggerTest(name)
         self = self@TestCaseInDir(name, ...
             fullfile(fileparts(which(mfilename)), 'helper_classes'));
      end
      
      function testTwoPassingTests(self)
         logger = TestRunLogger;
         suite = TestSuite('TwoPassingTests');
         suite.run(logger);
         
         assertEqual(logger.Log, ...
             {'TestRunStarted', ...
             'TestComponentStarted', ...
             'TestComponentStarted', 'TestComponentFinished', ...
             'TestComponentStarted', 'TestComponentFinished', ...
             'TestComponentFinished', ...
             'TestRunFinished'});
         
         assertEqual(logger.NumTestCases, 2);
         assertEqual(logger.NumFailures, 0);
         assertEqual(logger.NumErrors, 0);
         assertTrue(isempty(logger.Faults));
      end
      
      function testFailingTestCase(self)
         logger = TestRunLogger;
         suite = TestSuite('FailingTestCase');
         suite.run(logger);
         
         assertEqual(logger.Log, ...
             {'TestRunStarted', ...
             'TestComponentStarted', ...
             'TestComponentStarted', 'TestCaseFailure', 'TestComponentFinished', ...
             'TestComponentFinished', ...
             'TestRunFinished'});
         
         assertEqual(logger.NumTestCases, 1);
         assertEqual(logger.NumFailures, 1);
         assertEqual(logger.NumErrors, 0);
         assertEqual(numel(logger.Faults), 1);
         assertEqual(logger.Faults(1).Type, 'failure');
      end
      
   end

end