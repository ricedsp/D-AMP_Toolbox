function [] = assertVectorsAlmostEqual(a, b, comparetype, tolerance)
  if (max(abs(reshape(a-b,[],1))) > tolerance)
    testFailed(a,b);
  elseif (min(size(a) == size(b)) < 1)
    testFailed(a,b);
  end

function [] = testFailed(a, b)
  [ST, I] = dbstack(2);
  disp(strcat("FAILED: ",  ST(1).name));
  disp(a)
  disp("--")
  disp(b)
