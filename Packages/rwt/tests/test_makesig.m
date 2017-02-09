function test_suite = test_makesig
initTestSuite;

function test_makesig_heavisine
  x = makesig('HeaviSine', 8);
  y = [4.0000    0.0000   -6.0000   -2.0000    2.0000    0.0000   -4.0000   -0.0000];
assertVectorsAlmostEqual(x, y, 'relative', 0.0001);

function test_makesig_bumps
  x = makesig('Bumps', 8);
  y = [0.3206    5.0527    0.3727    0.0129    0.0295    0.0489    0.0004    0.0000];
assertVectorsAlmostEqual(x, y, 'relative', 0.0001);

function test_makesig_blocks
  x = makesig('Blocks', 8);
  y = [4.0000    0.5000    3.0000    0.9000    0.9000    5.2000   -0.0000   -0.0000];
assertVectorsAlmostEqual(x, y, 'relative', 0.0001);

function test_makesig_doppler
  x = makesig('Doppler', 12);
  y = [-0.1954 -0.3067 0.0000 -0.4703 0.4930 -0.2703 -0.4127 0.1025 0.4001 0.3454 0.1425 0];
assertVectorsAlmostEqual(x, y, 'relative', 0.0001);

function test_makesig_ramp
  x = makesig('Ramp', 8);
  y = [0.1250    0.2500   -0.6250   -0.5000   -0.3750   -0.2500   -0.1250         0];
assertVectorsAlmostEqual(x, y, 'relative', 0.0001);

function test_makesig_cusp
  x = makesig('Cusp', 8);
  y = [0.4950    0.3464    0.0707    0.3606    0.5050    0.6164    0.7106    0.7937];
assertVectorsAlmostEqual(x, y, 'relative', 0.0001);

function test_makesig_sing
  x = makesig('Sing', 8);
  y = [5.3333   16.0000   16.0000    5.3333    3.2000    2.2857    1.7778    1.4545];
assertVectorsAlmostEqual(x, y, 'relative', 0.0001);

function test_makesig_hisine
  x = makesig('HiSine', 8);
  y = [0.8267   -0.9302    0.2200    0.6827   -0.9882    0.4292    0.5053   -0.9977];
assertVectorsAlmostEqual(x, y, 'relative', 0.0001);

function test_makesig_losine
  x = makesig('LoSine', 8);
  y = [0.865973039158459   0.866130104544730   0.000314159260191  -0.865815888304075  -0.866287084447387  -0.000628318489377   0.865658651997088   0.866443978850937];
assertVectorsAlmostEqual(x, y, 'relative', 0.0000001);

function test_makesig_linchirp
  x = makesig('LinChirp', 8);
  y = [0.0491    0.1951    0.4276    0.7071    0.9415    0.9808    0.6716    0.0000];
assertVectorsAlmostEqual(x, y, 'relative', 0.0001);

function test_makesig_twochirp
  x = makesig('TwoChirp', 8);
  y = [0.5132    1.5000    0.5412    0.8660   -0.5132         0    0.5132    0.8660];
assertVectorsAlmostEqual(x, y, 'relative', 0.0001);

function test_makesig_quadchirp
  x = makesig('QuadChirp', 8);
  y = [0.0164    0.1305    0.4276    0.8660    0.8895   -0.3827   -0.6217    0.8660];
assertVectorsAlmostEqual(x, y, 'relative', 0.0001);

function test_makesig_mishmash
  x = makesig('MishMash', 8);
  y = [0.8922   -0.6046    1.0751    2.2558    0.8429    1.0273    0.5551   -0.1317];
assertVectorsAlmostEqual(x, y, 'relative', 0.0001);

function test_makesig_wernersorrows
  x = makesig('WernerSorrows', 8);
  y = [1.5545    5.3175    0.8252    1.6956   -1.2678    0.6466    1.7332   -0.9977];
assertVectorsAlmostEqual(x, y, 'relative', 0.0001);

function test_makesig_leopold
  x = makesig('Leopold', 8);
  y = [0     1     0     0     0     0     0     0];
assertVectorsAlmostEqual(x, y, 'relative', 0.0001);


