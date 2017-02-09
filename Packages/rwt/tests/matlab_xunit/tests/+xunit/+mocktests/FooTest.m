classdef FooTest < TestCase
    methods
        function object = FooTest(name)
            object = object@TestCase(name);
        end
        function test_sanity(object)
            assertEqual(0, 0)
        end
    end
end