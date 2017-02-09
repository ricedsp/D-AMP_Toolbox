function test_suite = test_comparisonMessage
%test_comparisonMessage Unit test for comparisonMessage.

%   Steven L. Eddins
%   Copyright 2009 The MathWorks, Inc.

initTestSuite;

function test_happyCase
s = xunit.utils.comparisonMessage('user message', 'assertion message', ...
    [1 2 3], 'hello');
c = xunit.utils.stringToCellArray(s);

expected_output = { 'user message' 
    'assertion message' 
    ''
    'First input:'
    '     1     2     3'
    ''
    'Second input:'
    'hello'};

assertEqual(c, expected_output);

    