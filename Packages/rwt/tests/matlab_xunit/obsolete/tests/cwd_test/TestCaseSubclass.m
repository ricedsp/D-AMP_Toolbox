%TestCaseSubclass TestCase subclass containing two passing tests

%   Steven L. Eddins
%   Copyright 2008 The MathWorks, Inc.

classdef TestCaseSubclass < TestCase
   methods
       function self = TestCaseSubclass(name)
           self = self@TestCase(name);
       end
       
       function testA(self)
       end
       
       function testB(self)
           % Intentionally fail this test case.
           assertFalse(true);
       end
   end
end