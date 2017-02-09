function test_suite = testIsAlmostEqual
%testIsAlmostEqual Unit tests for isAlmostEqual

%   Steven L. Eddins
%   Copyright 2008 The MathWorks, Inc.

initTestSuite;

function testExactlyEqual
A = [1 2; 3 4];
B = [1 2; 3 4];
assertTrue(mtest.utils.isAlmostEqual(A, B));

function testDefaultTolerance
assertTrue(mtest.utils.isAlmostEqual(1, 1+10*eps));
assertFalse(mtest.utils.isAlmostEqual(1, 1+1000*eps));

function testDefaultToleranceSingle
assertTrue(mtest.utils.isAlmostEqual(single(1), 1 + 10*eps('single')));
assertFalse(mtest.utils.isAlmostEqual(single(1), 1 + 1000*eps('single')));

function testSpecifiedTolerance
assertTrue(mtest.utils.isAlmostEqual(1, 1.09, 0.1));
assertFalse(mtest.utils.isAlmostEqual(1, 1.2, 0.1));

function testSpecialValues
A = [Inf, -Inf, NaN, 2.0];
B = [Inf, -Inf, NaN, 2.0+10*eps];
assertTrue(mtest.utils.isAlmostEqual(A, B));

C = [Inf, -Inf, NaN, 2.0];
D = [Inf, -Inf, 0, 2.0+10*eps];
assertFalse(mtest.utils.isAlmostEqual(C, D));

function testUint8
assertTrue(mtest.utils.isAlmostEqual(uint8(1), uint8(1)));
assertFalse(mtest.utils.isAlmostEqual(uint8(1), uint8(2)));

function testChar
assertTrue(mtest.utils.isAlmostEqual('foobar', 'foobar'));
assertFalse(mtest.utils.isAlmostEqual('foo', 'bar'));