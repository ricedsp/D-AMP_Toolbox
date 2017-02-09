classdef WrongExceptionThrownTest < TestCase
   methods
      function self = WrongExceptionThrownTest(methodName)
         self = self@TestCase(methodName);
      end
      
      function testThrowsException(self)
         f = @() error('d:e:f', 'message');
         assertExceptionThrown(f, 'a:b:c');
      end
   end
end