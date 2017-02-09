classdef BadFixture < TestCase
    
    methods
        function self = BadFixture(name)
            self = self@TestCase(name);
        end
        
        function setUp(self)
            throw(MException('setUpError:BadFixture', ...
                'BadFixture setUp method always throws exception'));
        end
        
        function testMethod(self)
        end
    end
end