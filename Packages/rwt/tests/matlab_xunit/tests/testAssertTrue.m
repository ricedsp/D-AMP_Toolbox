function test_suite = testAssertTrue
%testAssertTrue Unit tests for assertTrue

%   Steven L. Eddins
%   Copyright 2008 The MathWorks, Inc.

initTestSuite;

function testAssertTrueHappyCase
assertTrue(true);

function testAssertTrueHappyCaseWithTwoArgs
assertTrue(true, '1.e4 e5 2.Nf3 Nc6');

function testAssertTrueFailed
% Verify exception when false is passed to assertTrue.
assertExceptionThrown(@() assertTrue(false), 'assertTrue:falseCondition');

function testAssertTrueNonscalar
% Verify that assertTrue doesn't like nonscalar input.
assertExceptionThrown(@() assertTrue(logical([1 1])), 'assertTrue:invalidCondition');

function testAssertTrueNonlogical
% Verify that assertTrue doesn't like nonlogical input.
assertExceptionThrown(@() assertTrue(5), 'assertTrue:invalidCondition');
