function suite = notTestString
% This function exists to help test that the TestSuite.fromPwd() method does not
% pick up function-handle test files that do not match the naming convention.
initTestSuite;

function testA

function testB


