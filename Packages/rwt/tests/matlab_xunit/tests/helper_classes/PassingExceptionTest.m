classdef PassingExceptionTest < TestCase
   methods
      function self = PassingExceptionTest(methodName)
         self = self@TestCase(methodName);
      end
      
      function testThrowsException(self)
         f = @() error('a:b:c', 'error message');
         assertExceptionThrown(f, 'a:b:c');
      end
   end
end