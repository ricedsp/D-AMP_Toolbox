function test_suite = testAssertEqual
%testAssertEqual Unit tests for assertEqual

%   Steven L. Eddins
%   Copyright 2008 The MathWorks, Inc.

initTestSuite;

function testAssertEqualHappyCase
assertEqual(5, 5);

function testAssertEqualWithThreeInputs
assertEqual(5, 5, 'Scandinavian Defense');

function testAssertEqualHappyCaseString
assertEqual('foobar', 'foobar');

function testAssertEqualHappyCaseMatrix
assertEqual(magic(3), magic(3))

function testInfAndInf
assertEqual(Inf, Inf);

function testMinusInfAndMinusInf
assertEqual(-Inf, -Inf);

function testOppositeSignInfs
assertExceptionThrown(@() assertEqual(-Inf, Inf), 'assertEqual:nonEqual');

function testFiniteAndInf
assertExceptionThrown(@() assertEqual(1, Inf), 'assertEqual:nonEqual');

function testFiniteAndNaN
assertExceptionThrown(@() assertEqual(1, NaN), 'assertEqual:nonEqual');

function testInfiniteAndNaN
assertExceptionThrown(@() assertEqual(Inf, NaN), 'assertEqual:nonEqual');

function testAssertEqualNotEqual
assertExceptionThrown(@() assertEqual(5, 4), 'assertEqual:nonEqual');

function testAssertEqualSparsity
assertExceptionThrown(@() assertEqual(5, sparse(5)), 'assertEqual:sparsityNotEqual');

function testAssertEqualNans
assertEqual([1 NaN 2], [1 NaN 2]);

function testAssertEqualClass
assertExceptionThrown(@() assertEqual(5, uint8(5)), 'assertEqual:classNotEqual');