function testFliplrMatrix
%testFliplrMatrix Unit test for fliplr with matrix input

in = magic(3);
assertEqual(fliplr(in), in(:, [3 2 1]));
