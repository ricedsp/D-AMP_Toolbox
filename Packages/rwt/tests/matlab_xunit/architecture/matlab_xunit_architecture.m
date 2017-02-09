%% MATLAB xUnit Test Framework: Architectural Notes
% This document summarizes the key classes and design choices for MATLAB xUnit,
% a MATLAB unit testing framework based on xUnit patterns.
%
% Note: Testing pattern and smell terminology in this document is drawn from
% _xUnit Test Patterns: Refactoring Test Code_, by Gerard Meszaros,
% Addison-Wesley, 2007.

%% TestComponent, TestCase, and TestSuite
%
% <<class_diagram_a.gif>>
%
% The abstract |TestComponent| class defines an object that has a description (a
% name and a location) and that can be run.
%
% A |TestCase| object is a test component that defines an individual test case
% that can be run with a pass or fail result.
%
% A |TestSuite| object is a test component that contains a collection of other
% test components.  Note the hierarchical nature of test suites; they can
% contain both individual test case objects as well as other test suites.
% Running a test suite means invoking the |run| method on each test component in
% its collection.

%% TestCase: The Four-Phase Test
%
% The TestCase class provides the standard xUnit _Four-Phase Test_, using
% a _Fresh Fixture_, _Implicit Setup_, and _Implicit Teardown_. These all
% elements can all be seen in the |run| method of TestCase:
%
%         function did_pass = run(self, monitor)
%             %run Execute the test case
%             %    test_case.run(monitor) calls the TestCase object's setUp()
%             %    method, then the test method, then the tearDown() method.
%             %    observer is a TestRunObserver object.  The testStarted(),
%             %    testFailure(), testError(), and testFinished() methods of
%             %    observer are called at the appropriate times.  monitor is a
%             %    TestRunMonitor object.  Typically it is either a TestRunLogger
%             %    subclass or a CommandWindowTestRunDisplay subclass.
%             %
%             %    test_case.run() automatically uses a
%             %    CommandWindowTestRunDisplay object in order to print test
%             %    suite execution information to the Command Window.
%             
%             if nargin < 2
%                 monitor = CommandWindowTestRunDisplay();
%             end
%             
%             did_pass = true;
%             monitor.testComponentStarted(self);
%             
%             try
%                 self.setUp();
%                 f = str2func(self.MethodName);
%                 
%                 try
%                     % Call the test method.
%                     f(self);
%                 catch failureException
%                     monitor.testCaseFailure(self, failureException);
%                     did_pass = false;
%                 end
%                 
%                 self.tearDown();
%                 
%             catch errorException
%                 monitor.testCaseError(self, errorException);
%                 did_pass = false;
%             end
%             
%             monitor.testComponentFinished(self, did_pass);
%         end
%
% Phase 1 sets up the test fixture via the _Implicit Setup_ call, |self.setUp()|.
% The base class |setUp()| method does nothing.
%
% Phases 2 and 3 (exercising the system under test and verifying the expected
% outcome) are handled by the test method, which is invoked by |f(self)|.
%
% Phase 4 tears down the test fixture via the _Implicit Teardown_ call,
% |self.tearDown()|.  The base class |tearDown()| method does nothing.
%
% Test failure and test error exceptions are caught and handled by the |run()|
% method, so test methods do not need to use try-catch.  This facilitates
% simple, straight-line test-method code.
%
% _Note: The |monitor| object will be discussed later._

%% Test Case Discovery
% The static method |TestSuite.fromName| constructs a test suite based on the
% name of an M-file.  If the M-file defines a |TestCase| subclass, then |fromName|
% inspects the methods of the class and constructs a |TestCase| object for each
% method whose name begins with "[tT]est".  If the M-file does not define a
% |TestCase| subclass, then |fromName| attempts to construct either a simple
% procedural test case or a set of subfunction-based test cases.  (See the next
% section).
%
% The static method |TestSuite.fromPwd| constructs a test suite by discovering
% all the test cases in the present working directory.  It discovers all
% |TestCase| subclasses in the directory. In addition, it constructs test suites
% from all the procedural M-files in the directory beginning with "[tT]est".
%
% The _File System Test Runner_, |runtests|, provides convenient syntaxes for
% performing test case discovery automatically.

%% FunctionHandleTestCase: For the Procedural World
% Most MATLAB users are much more comfortable with procedural programming.  An
% important design goal for MATLAB xUnit is to make it as easy as possible for MATLAB
% users with little object-oriented programming experience to create and run
% their own tests.  The FunctionHandleTestCase supplies the plumbing necessary
% to support procedural test functions:
%
% <<class_diagram_b.gif>>
%
% Private properties |SetupFcn|, |TestFcn|, and |TeardownFcn| are procedural 
% _function handles_ (similar to function pointers or function references in
% other languages).
%
% |runTestCase()| is the test method used for constructing a TestCase object.
%
% Managing test fixtures requires special consideration, because procedural
% function handles don't have access to object instance data in order to access
% a test fixture.
%
% The overridden |setUp()| method looks at the number of outputs of the function
% handle |SetupFcn|.  If it has an output argument, then the argument is saved
% in the private |TestData| property, and |TestData| is then passed to both
% |TestFcn| and |TeardownFcn| for their use.

%% Writing Procedural Test Cases
% Procedural test cases can be written in two ways: 
%
% * A simple M-file function that is treated as a single test case
% * An M-file containing multiple subfunctions that are each treated as a test case. 
%
% In either case, the test
% case is considered to pass if it executes without error.
%
% Writing one test case per file is not ideal; it would lead to either zillions
% of tiny little test files, or long test methods exhibiting various bad test
% smells (_Multiple Test Conditions_, _Flexible Test_, _Conditional Test Logic_,
% _Eager Test_, _Obscure Test_, etc.)  So we need a way to write multiple test
% cases in a single procedural M-file.  The natural MATLAB way would be to use
% subfunctions.
%
% However, subfunction-based test cases require special consideration.  Consider
% the following M-file structure:
%
%    === File A.m ===
%    function A
%       ...
% 
%    function B
%       ...
% 
%    function C
%       ...
% 
%    function D
%       ...
%
% The first function in the file, |A|, has the same name as the file.  When
% other code outside this function calls |A|, it is this first function that
% gets called.  Functions |B|, |C|, and |D| are called _subfunctions_.
% Normally, these subfunctions are only visible to and can only be called by
% |A|.  The only way that code elsewhere might be able to call |B|, |C|, or |D|
% is if function |A| forms handles to them and passes those handles out of its
% scope.  Normally this would be done by returning the function handles as
% output arguments.
%
% Note that no code executing outside the scope of a function in A.m can form
% function handles to |B|, |C|, or |D|, or can even determine that these
% functions exist.
%
% This obviously poses a problem for test discovery!
%
% The MATLAB xUnit solution is to establish the following convention for
% subfunction-based tests.  The first function in a test M-file containing
% subfunction tests has to begin with these lines:
%
%    === File A.m ===
%    function test_suite = A
%    initTestSuite;
%    ...
%
% |initTestSuite| is a _script_ that runs in the scope of the function |A|.
% |initTestSuite| determines which subfunctions are test functions, as well as setup
% or teardown functions.  It forms handles to these functions and constructs a
% set of FunctionHandleTestCase objects, which function |A| returns as the
% output argument |test_suite|.

%% TestRunMonitor
% The abstract |TestRunMonitor| class defines the interface for an object that
% "observe" the in-progress execution of a test suite.  MATLAB xUnit provides two
% subclasses of |TestRunMonitor|:
%
% * |TestRunLogger| silently logs test suite events and captures the details of
% any test failures or test errors.
% * |CommandWindowTestRunDisplay| prints the progress of an executing test suite
% to the Command Window.
%
% <<class_diagram_c.gif>>
%
% A TestRunMonitor is passed to the |run()| method of a TestComponent object.
% The |run()| method calls the appropriate notification methods of the
% monitor.
%
% Here is the output when using the CommandWindowTestRunDisplay object on the
% MATLAB xUnit's own test suite:
%
%    runtests 
%    Starting test run with 92 test cases.
%    ....................
%    ....................
%    ....................
%    ....................
%    ............
%    PASSED in 7.040 seconds.

%% File System Test Runner
% MATLAB xUnit provides a command-line _File System Test Runner_ called
% |runtests|.  When called with no input arguments, |runtests| gathers all the 
% test cases from the current directory and runs them, summarizing the results
% to the Command Window.  |runtests| can also take a string argument specifying
% which test file, and optionally which specific test case, to run.

%% Test Selection
% Test selection is supported in |runtests| by passing in a string of the form:
%
%     'Location:Name'
%
% or just:
%
%     'Location'
%
% Both of these forms are handled by |runtests| and by |TestSuite.fromName|.
%
% 'Location' is the name of the M-file containing test cases.  'Name' is the
% name of a specific test case.  Normally, the name of the test case is the name
% of the corresponding TestCase method.  For FunctionHandleTestCase objects,
% though, 'Name' is the subfunction name.

%% Assertion Methods
% MATLAB xUnit provides the following assertion methods:
%
% * _Stated Outcome Assertion_ (|assertTrue|, |assertFalse|)
% * _Equality Assertion_ (|assertEqual|)
% * _Fuzzy Equality Assertion_ (|assertElementsAlmostEqual|, |assertVectorsAlmostEqual|)
% * _Expected Exception Assertion_ (|assertExceptionRaised|)
%
% Assertion functions are provided via globally accessible names (e.g.,
% |assertEqual|).  The assertion functions could be moved to the |xunit|
% package, but MATLAB users are not accustomed yet to packages and package
% name-scoping syntax.
%
% 'message' is the last input to the assertion functions and is optional.  (See
% below for discussion of _Assertion Roulette_.)
%
% The _Expected Exception Assertion_, |assertExceptionRaised| is used by forming
% an anonymous function handle from an expression that is expected to error, and
% then passing that function handle to |assertExceptionRaised| along with the
% expected exception identifier. For example:
%
%    f = @() sin(1,2,3);
%    assertExceptionRaised(f, 'MATLAB:maxrhs')
%
% By using this mechanism, test writers can verify exceptions without using
% try-catch logic in their test code.

%% Stack Traces and "Assertion Roulette"
% _xUnit Test Patterns_ explains the smell _Assertion Roulette_ this way: "It is
% hard to tell which of several assertions within the same test method caused a
% test failure.
%
% MATLAB xUnit mitigates against _Assertion Roulette_ by capturing the entire stack
% trace, including line numbers, for every test failure and test error.  (The
% MATLAB MException object, which you obtain via the |catch| clause, contains
% the stack trace.)  The stack trace is displayed to the Command Window, with
% clickable links that load the corresponding M-file into editor at the
% appropriate line number.
%
% Stack traces can be pretty long, though.  Also, test framework plumbing tends
% to occupy the trace in between the assertion and the user's test code, thus
% making the trace hard to interpret for less-experienced users.  MATLAB xUnit,
% therefore, uses a stack filtering heuristic for displaying test fault traces:
% Starting at the deepest call level, once the trace leaves MATLAB xUnit framework
% functions, all further framework functions are filtered out of the stack
% trace.
%
% Here's an example of stack trace display in the output of |runtests|:
%
% <html>
% <tt>
% >> runtests testSample<br />
% Starting test run with 1 test case.<br />
% F<br />
% FAILED in 0.081 seconds.<br />
% <br />
% ===== Test Case Failure =====<br />
% Location: c:\work\matlab_xunit\architecture\testSample.m<br />
% Name:     testMyCode<br />
% <br />
% c:\work\matlab_xunit\architecture\testSample.m at <span style="color:blue; 
% text-decoration:underline">line 6</span><br />
% <br />
% Input elements are not all equal within relative tolerance: 1.49012e-008<br />
% <br />
% First input:<br />
%      1<br />
% <br />
% Second input:<br />
%     1.1000<br />
% </tt>
% </html>
%
% Clicking on the blue, underlined link above loads the corresponding file into
% the editor, positioned at the appropriate line.

%% Extending the Framework
% The MATLAB xUnit framework can be extended primarily by subclassing |TestCase|,
% |TestSuite|, and |TestMonitor|.
%
% |TestCase| can be subclassed to enable a new set of test cases that all share
% some particular behavior.  The MATLAB xUnit Test Framework contains three
% examples of extending |TestCase| behavior in this way:
%
% * |FunctionHandleTestCase| provides the ability to define test cases based on
% procedural function handles.
% * |TestCaseInDir| defines a test case that must be run inside a particular
% directory. The |setUp| and |tearDown| functions are overridden to change the
% MATLAB working directory before running the test case, and then to restore the
% original working directory when the test case finished.  The class is used by
% the framework's own test suite.
% * |TestCaseInPath| defines a test case that must be run with a particular
% directory temporarily added to the MATLAB path.  Its implementation is similar
% to |TestCaseInDir|, and it is also used by the framework's own test suite.
%
% |TestSuite| could be similarly extended by subclassing. This might a provide a
% way in the future to define a test suite containing collections of test
% components in separate directories, which is not currently supported.
%
% Finally |TestRunMonitor| could be subclassed to support a variety of test
% monitoring mechanisms, such as what might be required by a _Graphical Test
% Runner_.