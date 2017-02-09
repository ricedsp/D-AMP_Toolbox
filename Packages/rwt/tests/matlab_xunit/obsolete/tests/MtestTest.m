%TestSuiteTest Unit tests for mtest command-line test runner.

classdef MtestTest < TestCaseInDir

   methods
       
       function self = MtestTest(name)
           self = self@TestCaseInDir(name, ...
               fullfile(fileparts(which(mfilename)), 'cwd_test'));
       end
      
      function test_noInputArgs(self)
          [T, did_pass] = evalc('mtest');
          % The cwd_test directory contains some test cases that fail,
          % so output of mtest should be false.
          assertFalse(did_pass);
      end
      
      function test_oneInputArg(self)
          [T, did_pass] = evalc('mtest(''testFoobar'')');
          % cwd_test/testFoobar.m is supposed to pass.
          assertTrue(did_pass);
      end
      
      function test_oneInputArgWithFilter_passing(self)
          [T, did_pass] = evalc('mtest(''TestCaseSubclass:testA'')');
          assertTrue(did_pass);
      end
      
      function test_oneInputArgWithFilter_failing(self)
          [T, did_pass] = evalc('mtest(''TestCaseSubclass:testB'')');
          assertFalse(did_pass);
      end
      
   end

end