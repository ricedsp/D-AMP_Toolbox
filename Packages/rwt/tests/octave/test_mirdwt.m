function test_suite = test_mirdwt
  disp("mrdwt")
  test_mirdwt_1
  test_mirdwt_2

function test_mirdwt_1     
       xin = makesig('Leopold',8);
       h = daubcqf(4,'min');
       Lin = 1;
       [yl,yh,L] = mrdwt(xin,h,Lin);
       [x,L] = mirdwt(yl,yh,h,L);

assertEqual(L,Lin);
assertVectorsAlmostEqual(x, xin,'relative',0.0001);

function test_mirdwt_2
       load ../lena512; 
       x = lena512;
       h = daubcqf(6);
       [yl,yh,L] = mrdwt(x,h);
assertEqual(L,9);
       [x_new,L] = mirdwt(yl,yh,h);
assertEqual(L,9);
assertVectorsAlmostEqual(x, x_new,'relative',0.0001);
