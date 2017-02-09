function test_suite = testAssertAlmostEqual
%testAssertAlmostEqual Unit tests for assertAlmostEqual

%   Steven L. Eddins
%   Copyright 2008 The MathWorks, Inc.

initTestSuite;

function testEqual
assertAlmostEqual(1, 1);

function testEqualWithThreeInputs
assertAlmostEqual(1, 1.1, 0.2);

function testEqualWithFourInputs
assertExceptionThrown(@() assertAlmostEqual(1, 2, 0.1, 'checkmate'), ...
    'assertAlmostEqual:tolExceeded');

function testEmptyRelTol
assertAlmostEqual(1, 1+10*eps, [], 'checkmate');

function testNotEqual
assertExceptionThrown(@() assertAlmostEqual(1, 1+1000*eps), ...
    'assertAlmostEqual:tolExceeded');

function testSingleEqual
assertAlmostEqual(single(1), single(1 + 10*eps('single')));

function testSingleNotEqual
assertExceptionThrown(@() assertAlmostEqual(single(1), ...
    single(1 + 1000*eps('single'))), 'assertAlmostEqual:tolExceeded');

function testZeros
assertAlmostEqual(0, 0);

function testSingleZeros
assertAlmostEqual(single(0), single(0));

function testSparse
assertAlmostEqual(sparse(1), sparse(1 + 10*eps));
