function test_suite = testFunctionHandlesD
%testFunctionHandlesD Test file used by TestFunctionHandlesTest
%   Contains two passing tests that use a test fixture with no test data.

%   Steven L. Eddins
%   Copyright 2008 The MathWorks, Inc.

initTestSuite;

function setUpFcn


function testA(varargin)
assertTrue(isempty(varargin));

function testB(varargin)
assertTrue(isempty(varargin));

function tearDownFcn(varargin)
assertTrue(isempty(varargin));

