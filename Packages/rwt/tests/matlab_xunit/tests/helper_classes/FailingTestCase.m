% FailingTestCase
% Utility class used by unit tests.

% Steven L. Eddins
% Copyright 2008 The MathWorks, Inc.

classdef FailingTestCase < TestCase

   methods
      function self = FailingTestCase(name)
         self = self@TestCase(name);
      end

      function testFail(self)
         throw(MException('testFail:FailingTestCase', ...
            'testFail always fails'));
      end
   end

end
