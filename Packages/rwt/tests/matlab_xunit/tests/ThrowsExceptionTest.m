classdef ThrowsExceptionTest < TestCaseInDir
    
    methods
        function self = ThrowsExceptionTest(methodName)
            self = self@TestCaseInDir(methodName, ...
                fullfile(fileparts(which(mfilename)), 'helper_classes'));
        end
        
        function testPassingTest(self)
            logger = TestRunLogger();
            TestSuite('PassingExceptionTest').run(logger);
            assertTrue((logger.NumTestCases == 1) && ...
                (logger.NumFailures == 0) && ...
                (logger.NumErrors == 0), ...
                'Passing exception test should have no failures or errors');
        end
        
        function testNoExceptionTest(self)
            logger = TestRunLogger();
            TestSuite('ExceptionNotThrownTest').run(logger);
            assertTrue(strcmp(logger.Faults(1).Exception.identifier, ...
                'assertExceptionThrown:noException'), ...
                'Fault exception should be throwsException:noException');
        end
        
        function testWrongExceptionTest(self)
            logger = TestRunLogger();
            TestSuite('WrongExceptionThrownTest').run(logger);
            assertTrue(strcmp(logger.Faults(1).Exception.identifier, ...
                'assertExceptionThrown:wrongException'), ...
                'Fault exception should be throwsException:wrongException');
        end
        
    end
    
    
end