function test_suite = test_setopt
  disp("setopt")
  test_setopt_all_defaults
  test_setopt_nonzero_becomes_zero

function test_setopt_all_defaults
  x            = [];
  default_opts = [5 6 7 8];
  z = setopt(x, default_opts);
  z_corr       = [5 6 7 8];
assertVectorsAlmostEqual(z, z_corr, 'relative', 0.0001);

function test_setopt_nonzero_becomes_zero
  x            = [1 0 3];
  default_opts = [5 6 7 8];
  z = setopt(x, default_opts);
  z_corr       = [1 6 3 8];
  %z_corr       = [1 0 3 8];   % This would be more intuitive 
assertVectorsAlmostEqual(z, z_corr, 'relative', 0.0001);
