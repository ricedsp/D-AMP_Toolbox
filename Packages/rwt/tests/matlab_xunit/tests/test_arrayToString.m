function test_suite = test_arrayToString
%test_arrayToString Unit test for arrayToString.

%   Steven L. Eddins
%   Copyright 2009 The MathWorks, Inc.

initTestSuite;

function test_smallInput
A = [1 2 3];
assertEqual(strtrim(xunit.utils.arrayToString(A)), '1     2     3');

function test_largeInput
A = zeros(1000, 1000);
assertEqual(xunit.utils.arrayToString(A), '[1000x1000 double]');

function test_emptyInput
assertEqual(xunit.utils.arrayToString(zeros(1,0,2)), '[1x0x2 double]');
