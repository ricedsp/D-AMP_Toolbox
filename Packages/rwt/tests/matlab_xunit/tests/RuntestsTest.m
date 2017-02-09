%TestSuiteTest Unit tests for runtests command-line test runner.

classdef RuntestsTest < TestCaseInDir

   methods
       
       function self = RuntestsTest(name)
           self = self@TestCaseInDir(name, ...
               fullfile(fileparts(which(mfilename)), 'cwd_test'));
       end
      
      function test_noInputArgs(self)
          [T, did_pass] = evalc('runtests');
          % The cwd_test directory contains some test cases that fail,
          % so output of runtests should be false.
          assertFalse(did_pass);
      end
      
      function test_Verbose(self)
          [T, did_pass] = evalc('runtests(''-verbose'')');
          assertFalse(did_pass);
      end
      
      function test_oneInputArg(self)
          [T, did_pass] = evalc('runtests(''testFoobar'')');
          % cwd_test/testFoobar.m is supposed to pass.
          assertTrue(did_pass);
      end
      
      function test_verboseThenTestName(self)
          [T, did_pass] = evalc('runtests(''-verbose'', ''.'')');
          assertFalse(did_pass);
      end
      
      function test_testNameThenVerbose(self)
          [T, did_pass] = evalc('runtests(''.'', ''-verbose'')');
          assertFalse(did_pass);
      end
      
      function test_oneInputArgWithFilter_passing(self)
          [T, did_pass] = evalc('runtests(''TestCaseSubclass:testA'')');
          assertTrue(did_pass);
      end
      
      function test_oneInputArgWithFilter_failing(self)
          [T, did_pass] = evalc('runtests(''TestCaseSubclass:testB'')');
          assertFalse(did_pass);
      end
      
      function test_oneDirname(self)
          [T, did_pass] = evalc('runtests(''../dir1'')');
          assertTrue(did_pass);
          
          [T, did_pass] = evalc('runtests(''../dir2'')');
          assertFalse(did_pass);
      end
      
      function test_twoDirnames(self)
          [T, did_pass] = evalc('runtests(''../dir1'', ''../dir2'')');
          assertFalse(did_pass);
      end
      
      function test_packageName(self)
          [T, did_pass] = evalc('runtests(''xunit.mocktests'')');
          assertTrue(did_pass);
      end
      
      function test_noTestCasesFound(self)
          assertExceptionThrown(@() runtests('no_such_test'), ...
              'xunit:runtests:noTestCasesFound');
      end
      
      function test_optionStringsIgnored(self)
          % Option string at beginning.
          [T, did_pass] = evalc('runtests(''-bogus'', ''../dir1'')');
          assertTrue(did_pass);
          
          % Option string at end.
          [T, did_pass] = evalc('runtests(''../dir2'', ''-bogus'')');
          assertFalse(did_pass);
      end
      
      function test_logfile(self)
          name = tempname;
          command = sprintf('runtests(''../dir1'', ''-logfile'', ''%s'')', name);
          [T, did_pass] = evalc(command);
          assertTrue(did_pass);
          assertTrue(exist(name, 'file') ~= 0);
          delete(name);
      end
      
      function test_logfileWithNoFile(self)
          assertExceptionThrown(@() runtests('../dir1', '-logfile'), ...
              'xunit:runtests:MissingLogfile');
      end
      
      function test_logfileWithNoWritePermission(self)
          assertExceptionThrown(@() runtests('../dir1', '-logfile', ...
              'C:\dir__does__not__exist\foobar.txt'), ...
              'xunit:runtests:FileOpenFailed');
      end
      
      function test_namesInCellArray(self)
          [T, did_pass] = evalc('runtests({''TestCaseSubclass:testA''})');
          assertTrue(did_pass);
          
          [T, did_pass] = evalc('runtests({''TestCaseSubclass:testA'', ''TestCaseSubclass:testB''})');
          assertFalse(did_pass);
      end
      
   end

end