function test_suite = testIsSetUpString
%testIsSetUpString Unit tests for isSetUpString

%   Steven L. Eddins
%   Copyright 2008 The MathWorks, Inc.

initTestSuite;

function testOneStringIs
assertTrue(xunit.utils.isSetUpString('setup'));
assertTrue(xunit.utils.isSetUpString('setUp_fixture'));

function testOneStringIsNot
assertFalse(xunit.utils.isSetUpString('bogus'));

function testCellArray
strs = {'setup', 'bogus'};
assertEqual(xunit.utils.isSetUpString(strs), [true false]);
assertEqual(xunit.utils.isSetUpString(strs'), [true; false]);