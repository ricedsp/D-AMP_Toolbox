% LoggingTestCase
% Utility class used by unit tests.

% Steven L. Eddins
% Copyright 2008 The MathWorks, Inc.

classdef LoggingTestCase < TestCase
    
    properties
        log = {};
    end
    
    methods
        function self = LoggingTestCase(name)
            self = self@TestCase(name);
        end
        
        function setUp(self)
            self.log{end + 1} = 'setUp';
        end
        
        function tearDown(self)
            self.log{end + 1} = 'tearDown';
        end
        
        function testMethod(self)
            self.log{end + 1} = 'testMethod';
        end
        
        function testBrokenMethod(self)
            throw(MException('brokenMethod:WasRun', ...
                'Call to testBrokenMethod always throws exception'));
        end
    end
    
end
