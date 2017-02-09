function test_suite = testIsTearDownString
%testIsTearDownString Unit tests for isTearDownString

%   Steven L. Eddins
%   Copyright 2008 The MathWorks, Inc.

initTestSuite;

function testOneStringIs
assertTrue(xunit.utils.isTearDownString('teardownfoobar'));
assertTrue(xunit.utils.isTearDownString('TearDown_foobar'));

function testOneStringIsNot
assertFalse(xunit.utils.isTearDownString('tEardown'));

function testCellArray
strs = {'teardown', 'tearup'};
assertEqual(xunit.utils.isTearDownString(strs), [true false]);
assertEqual(xunit.utils.isTearDownString(strs'), [true; false]);