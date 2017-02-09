function test_suite = testFunctionHandlesA
%testFunctionHandlesE Test file used by TestFunctionHandlesTest
%   Contains one failing test.

%   Steven L. Eddins
%   Copyright 2008 The MathWorks, Inc.

initTestSuite;

function testA
error('testFunctionHandlesA:expectedFailure', 'Bogus message');

