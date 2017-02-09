function suite = testFunctionHandlesTeardownNoSetup
% Verify that test file works if it has a teardown function but no setup
% function.
initTestSuite;

function teardown
close all

function test_normalCase
assertEqual(1, 1);

