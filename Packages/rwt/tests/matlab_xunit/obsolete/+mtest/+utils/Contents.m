% UTILS Utility package for MTEST unit testing framework
%
% Array Comparison
%   compareFloats            - Compare floating-point arrays using tolerance
%
% Test Case Discovery Functions
%   isTestCaseSubclass       - True for name of TestCase subclass
%
% String Functions
%   containsRegexp           - True if string contains regular expression
%   isSetUpString            - True for string that looks like a setup function
%   isTearDownString         - True for string that looks like teardown function
%   isTestString             - True for string that looks like a test function
%
% Miscellaneous Functions
%   generateDoc              - Publish test scripts in mtest/doc
%   parseFloatAssertInputs   - Common input-parsing logic for several functions

% Undocumented Functions
%   isAlmostEqual        - Floating-point equality test using relative tolerance

%   Steven L. Eddins
%   Copyright 2008-2009 The MathWorks, Inc.

