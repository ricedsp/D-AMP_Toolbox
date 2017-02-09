function test_suite = test_stringToCellArray
%test_stringToCellArray Unit test for stringToCellArray

%   Steven L. Eddins
%   Copyright 2009 The MathWorks, Inc.

initTestSuite;

function test_happyCase
s = sprintf('Hello\nWorld');
assertEqual(xunit.utils.stringToCellArray(s), {'Hello' ; 'World'});

function test_emptyInput
assertEqual(xunit.utils.stringToCellArray(''), cell(0, 1));

function test_spacesInFront
s = sprintf('    Hello\n  World\n');
assertEqual(xunit.utils.stringToCellArray(s), {'    Hello' ; '  World'});

function test_spacesAtEnd
s = sprintf('Hello  \nWorld     ');
assertEqual(xunit.utils.stringToCellArray(s), {'Hello  ' ; 'World     '});

