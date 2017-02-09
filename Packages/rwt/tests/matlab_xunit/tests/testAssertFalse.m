function test_suite = testAssertFalse
%testAssertFalse Unit tests for assertFalse

%   Steven L. Eddins
%   Copyright 2008 The MathWorks, Inc.

initTestSuite;

function testAssertFalseHappyCase
assertFalse(false);

function testAssertFalseHappyCaseWithTwoArgs
assertFalse(false, '1.e4 e5 2.Nf3 Nc6');

function testAssertFalseFailed
% Verify exception when false is passed to assertFalse.
assertExceptionThrown(@() assertFalse(true), 'assertFalse:trueCondition');

function testAssertFalseNonscalar
% Verify that assertFalse doesn't like nonscalar input.
assertExceptionThrown(@() assertFalse(logical([0 0])), 'assertFalse:invalidCondition');

function testAssertFalseNonlogical
% Verify that assertFalse doesn't like nonlogical input.
assertExceptionThrown(@() assertFalse(0), 'assertFalse:invalidCondition');
