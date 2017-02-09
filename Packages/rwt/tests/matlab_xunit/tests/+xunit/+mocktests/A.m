% Class A is a TestCase subclass containing two test cases (test_a and test_b).
classdef A < TestCase
    
    methods
        function self = A(name)
            self = self@TestCase(name);
        end
        
        function test_a(self)
        end
        
        function test_b(self)
        end
    end
    
end