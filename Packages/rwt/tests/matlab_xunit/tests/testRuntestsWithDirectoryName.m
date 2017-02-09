function test_suite = testRuntestsWithDirectoryName
%testRuntestsWithDirectoryName Unit test for mtest('dirname') syntax.

initTestSuite;

function testDirName
current_dir = pwd;
target_dir = fullfile(fileparts(which(mfilename)), 'cwd_test');
[T, did_pass] = evalc('runtests(target_dir)');
assertFalse(did_pass);
assertEqual(current_dir, pwd);
