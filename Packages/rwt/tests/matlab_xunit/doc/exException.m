%% <../index.html MATLAB xUnit Test Framework>: How to Test an Error Message
% It's surprising to most people (but not quality engineers) how
% often programmers make errors in error-handling code.  Because of
% this unfortunate truth, it is useful to write unit tests that
% verify that your MATLAB code throws the proper error, at the
% proper time.
%
% The assertion function that makes this task easy is
% |assertExceptionThrown|.  This example shows how to write a unit
% test that verifies the "Too many input arguments" error for the
% |cos| function.
%
% Your first step is to determine the _error identifier_ associated
% with the error message.  You can find out the error identifier by
% using the |lasterror| function.
%
% If you call |cos| with two input arguments, like this:
%
%   cos(1, 2)
%
% you get this error message:
%
%   Error using ==> cos
%   Too many input arguments. 
%
% Then if you call |lasterror|, you get this output:
%
%   ans = 
%   
%          message: [1x45 char]
%       identifier: 'MATLAB:maxrhs'
%            stack: [0x1 struct]
%
% So the _identifier_ associated with this error message is
% |'MATLAB:maxrhs'|.
%
% When you write your test function, you'll form an anonymous
% function handle that calls |cos| with the erroneous additional
% input argument.

f = @() cos(1, 2)

%%
% You then pass this function to |assertExceptionThrown|, along with
% the expected error identifier.

assertExceptionThrown(f, 'MATLAB:maxrhs');

%%
% |assertExceptionThrown| verifies that when |f()| is called, an
% error results with the specified error identifier.
%
% Here's our error condition test for the |cos| function.

cd examples_general
type testCos

%%
% Run the test using |runtests|.

runtests testCos

%%
% <../index.html Back to MATLAB xUnit Test Framework>

%%
% Copyright 2008-2010 The MathWorks, Inc.