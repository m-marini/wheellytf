function P = lin_reg(Y, varargin)
  avg = mean(Y);
  if length(varargin) > 0
    X = varargin{1};
  else
    X = [1 : size(Y, 1)];
  endif
  m = cov(X, Y) / var(X);
  q = avg - m * mean(X);
  P = [q, m];
endfunction