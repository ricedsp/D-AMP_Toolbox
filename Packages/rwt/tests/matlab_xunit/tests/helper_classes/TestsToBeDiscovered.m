classdef TestsToBeDiscovered < TestCase

   methods
      function self = TestsToBeDiscovered(name)
         self = self@TestCase(name);
      end
      
      function testMethodA
      end
      
      function testMethodB
      end
      
      function notATestMethod
      end

   end

end