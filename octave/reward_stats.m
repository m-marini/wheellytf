clear all
file = "../dqn-20220801-201540.csv";
k = 1. / 30;

function Z = discount(X, Y, k)
  DT = -X(2 : end) + X(1 : end - 1);
  A = exp(DT * k);
  B = 1 - A;
  Z = Y;
  Z(2 : end) = B .* Y(2 : end);
  for i = 2 : size(Z)(1)
    Z(i) = Z(i) + A(i - 1) * Z(i - 1);
  endfor
endfunction

X = csvread(file);
Y = X(:, 2);
X = X(:, 1);

avg = mean(Y);
m = cov(X, Y) / var(X);
q = avg - m * mean(X);

Z = discount(X, Y, k);
R = m * X + q;

subplot(1, 2, 1)
plot(X, Z, X, R);
grid on

subplot(1, 2, 2)
hist(Y, 21)
grid on
