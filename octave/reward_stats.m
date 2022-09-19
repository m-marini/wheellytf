clear all
file = "../data/reward.csv";
k = 0.995;
Y = csvread(file);
X = [1 : size(Y, 1)];
avg = mean(Y);
P = lin_reg(Y, X);
R = polynomial(X, P);
Z = discount(Y, k);

printf("Average: %g\n", avg);
printf("Trend:   %g\n", P(2));

subplot(1, 2, 1)
plot(X, Z, X, R);
grid on

subplot(1, 2, 2)
hist(Y, 21)
grid on

