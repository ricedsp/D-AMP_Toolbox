%% <../index.html MATLAB xUnit Test Framework>: How to Write xUnit-Style Tests by Subclassing TestCase
% The MATLAB xUnit architecture is based closely on the xUnit style, in
% which each test case is an instance of a subclass of the base
% TestCase class.  Programmers who are familiar with this style may
% want to write their own TestCase subclasses instead of using
% <./exSubfunctionTests.html subfunction-based tests>.
%
% This example shows a TestCase subclass containing test case
% methods and test fixture methods.  If you are not familiar with
% defining your own classes in MATLAB, you might want to review the
% MATLAB documentation on 
% <http://www.mathworks.com/access/helpdesk/help/techdoc/matlab_oop/ug_intropage.html 
% classes and object-oriented programming>,
% or you can simply stick to using subfunction-based tests.
%
% The sample M-file begins with the |classdef| statement, which sets
% the name of the class and indicates that it is a subclass of
% |TestCase|.

cd examples_general
dbtype TestUsingTestCase 1

%%
% The properties block contains a field that is initialized by the
% setup method and is used by the two test methods.

dbtype TestUsingTestCase 3:5

%%
% The first method in the methods block is the constructor.  It
% takes the desired test method name as its input argument, and it
% passes that input along to the base class constructor.

dbtype TestUsingTestCase 7:10

%%
% The |setUp| method creates a figure window and stores its handle in
% the field |fh|.

dbtype TestUsingTestCase 12:14

%%
% Test methods are those beginning with "test".

dbtype TestUsingTestCase 20:26

%%
% The |tearDown| method cleans up by deleting the figure window.

dbtype TestUsingTestCase 16:18

%%
% Run the test cases in the class by calling |runtests| with the name
% of the class.

runtests TestUsingTestCase

%%
% <../index.html Back to MATLAB xUnit Test Framework>

%%
% Copyright 2008-2010 The MathWorks, Inc.