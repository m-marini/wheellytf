function Y=discount(X, ALPHA)
  N = size(X, 1);
  Y = zeros(size(X));
  Y(1) = X(1) * (1 - ALPHA);
  for I = [2 : N]
    Y(I) = X(I) * (1 - ALPHA) + Y(I - 1) * ALPHA;
  endfor
endfunction
