function test_suite = test_assertFilesEqual
%test_assertFilesEqual Unit test for assertFilesEqual

%   Steven L. Eddins
%   Copyright 2009 The MathWorks, Inc.

initTestSuite;

function test_equal
assertFilesEqual('black.tif', 'black.tif');

function test_differentSize
assertExceptionThrown(@() assertFilesEqual('black.tif', 'black.png'), ...
    'assertFilesEqual:sizeMismatch');

function test_sameSizeButDifferent
assertExceptionThrown(@() assertFilesEqual('black.tif', 'almost_black.tif'), ...
    'assertFilesEqual:valuesDiffer');

function test_oneFileEmpty
assertExceptionThrown(@() assertFilesEqual('empty_file', 'black.png'), ...
    'assertFilesEqual:sizeMismatch');

function test_bothFilesEmpty
assertFilesEqual('empty_file', 'empty_file');

function test_cannotReadFirstFile
assertExceptionThrown(@() assertFilesEqual('bogus', 'black.png'), ...
    'assertFilesEqual:readFailure');

function test_cannotReadSecondFile
assertExceptionThrown(@() assertFilesEqual('black.png', 'bogus'), ...
    'assertFilesEqual:readFailure');
