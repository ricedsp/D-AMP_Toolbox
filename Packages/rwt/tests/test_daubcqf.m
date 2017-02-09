function test_suite = test_daubcqf
initTestSuite;

function test_daubcqf_min
  [a, b] = daubcqf(4);
  ax = [0.482962913144534   0.836516303737808   0.224143868042013  -0.129409522551260];
  bx = [0.129409522551260   0.224143868042013  -0.836516303737808   0.482962913144534];
assertVectorsAlmostEqual(a, ax, 'relative', 0.001);
assertVectorsAlmostEqual(b, bx, 'relative', 0.001);

function test_daubcqf_max
  [a, b] = daubcqf(4, 'max');
  ax = [-0.129409522551260   0.224143868042013   0.836516303737808   0.482962913144534];
  bx = [-0.482962913144534   0.836516303737808  -0.224143868042013  -0.129409522551260];
assertVectorsAlmostEqual(a, ax, 'relative', 0.001);
assertVectorsAlmostEqual(b, bx, 'relative', 0.001);

function test_daubcqf_mid_even_k
  [a, b] = daubcqf(4, 'mid');
  ax = [0.482962913144534   0.836516303737808   0.224143868042013  -0.129409522551260];
  bx = [0.129409522551260   0.224143868042013  -0.836516303737808   0.482962913144534];
assertVectorsAlmostEqual(a, ax, 'relative', 0.001);
assertVectorsAlmostEqual(b, bx, 'relative', 0.001);

function test_daubcqf_mid_odd_k
  [a, b] = daubcqf(6, 'mid');
  ax = [0.332670552950083   0.806891509311093   0.459877502118491  -0.135011020010255  -0.085441273882027   0.035226291885710];
  bx = [-0.035226291885710  -0.085441273882027   0.135011020010255   0.459877502118491 -0.806891509311093   0.332670552950083];
assertVectorsAlmostEqual(a, ax, 'relative', 0.001);
assertVectorsAlmostEqual(b, bx, 'relative', 0.001);

function test_daubcqf_odd
  handle = @() daubcqf(9);
assertExceptionThrown(handle, '');
