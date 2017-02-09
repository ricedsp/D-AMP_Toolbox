function testFliplrVector
%testFliplrVector Unit test for fliplr with vector input

in = [1 4 10];
out = fliplr(in);
expected_out = [10 4 1];

if ~isequal(out, expected_out)
    error('testFliplrVector:notEqual', 'Incorrect output for vector.');
end
