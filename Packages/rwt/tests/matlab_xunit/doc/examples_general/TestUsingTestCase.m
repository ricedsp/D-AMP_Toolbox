classdef TestUsingTestCase < TestCase

    properties
        fh
    end

    methods
        function self = TestUsingTestCase(name)
            self = self@TestCase(name);
        end

        function setUp(self)
            self.fh = figure;
        end

        function tearDown(self)
            delete(self.fh);
        end

        function testColormapColumns(self)
            assertEqual(size(get(self.fh, 'Colormap'), 2), 3);
        end

        function testPointer(self)
            assertEqual(get(self.fh, 'Pointer'), 'arrow');
        end
    end
end
