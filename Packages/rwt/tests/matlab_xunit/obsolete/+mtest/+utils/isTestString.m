function tf = isTestString(str)
%isTestString True if string looks like the name of a test
%   tf = isTestString(str) returns true if the string str looks like the name of
%   a test.  If str is a cell array of strings, then isTestString tests each
%   string in the cell array, returning the results in a logical array with the
%   same size as str.

%   Steven L. Eddins
%   Copyright 2008-2009 The MathWorks, Inc.

test_exp = '^[tT]est';
tf = mtest.utils.containsRegexp(str, test_exp);
