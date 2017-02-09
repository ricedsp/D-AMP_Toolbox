function [] = assertEqual(a, b)
  if (a != b)
    testFailed;
  end

function [] = testFailed()
  [ST, I] = dbstack(2);
  disp(strcat("FAILED: ",  ST(1).name));
