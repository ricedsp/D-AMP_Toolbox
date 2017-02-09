function test_suite = test_midwt
initTestSuite;



function test_midwt_1D
       x = makesig('LinChirp',8);
       h = daubcqf(4,'min');
       L = 2;
       [y,L] = mdwt(x,h,L);
       [x_new,L] = midwt(y,h,L);
assertVectorsAlmostEqual(x, x_new,'relative',0.0001);

function test_midwt_2D
       load lena512; 
       x = lena512;
       h = daubcqf(6);
       [y,L] = mdwt(x,h);
       [x_new,L] = midwt(y,h);
assertEqual(L,9);
assertVectorsAlmostEqual(x, x_new,'relative',0.0001);


