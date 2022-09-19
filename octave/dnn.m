clear all;
FROM = 10000;
ALPHA=0.95;
REWARDS=csvread("../data/reward.csv");
AVG_REWARDS=csvread("../data/avg_reward.csv");
DELTA=csvread("../data/delta.csv");
PIS=csvread("../data/pi.csv");
V0=csvread("../data/v0.csv");
V1=csvread("../data/v1.csv");
S0=csvread("../data/s0.csv");
S1=csvread("../data/s1.csv");
ACTIONS=csvread("../data/actions.csv");
TRAINED_AVG_REWARDS=csvread("../data/trained_avg_reward.csv");
TRAINED_PIS=csvread("../data/trained_pi.csv");
TRAINED_V0=csvread("../data/trained_v0.csv");
TRAINED_V1=csvread("../data/trained_v1.csv");
GRAD_PIS=csvread("../data/grad_pi.csv");

N = size(REWARDS, 1);
FROM1 = min(N - 10, FROM);
RANGE = [FROM1 : FROM1 + 10];

printf("============================================\n");
for i = RANGE
  printf("Step %d\n", i);
  printf("  s(t):           %d\n", S0(i));
  printf("  Action:         %d\n", ACTIONS(i));
  printf("  Reward:         %f\n", REWARDS(i));
  printf("  s(t+1):         %d\n", S1(i));
  printf("  v(s(t))         %f\n", V0(i));
  printf("  v(s(t+1))       %f\n", V1(i));
  printf("  avg reward(t)   %f\n", AVG_REWARDS(i));
  printf("  delta           %f\n", DELTA(i));
  printf("  Probabilities:  ")
  printf("%f ", PIS(i, :));
  printf("\n");
  printf("  Grad prob:      ")
  printf("%f ", GRAD_PIS(i, :));
  printf("\n");
  printf("  v'(s(t))        %f\n", TRAINED_V0(i));
  printf("  v'(s(t+1))      %f\n", TRAINED_V1(i));
  printf("  Probabilities': ")
  printf("%f ", TRAINED_PIS(i, :));
  printf("\n");
  printf("  avg reward(t+1) %f\n", TRAINED_AVG_REWARDS(i));
  printf("\n");
endfor

printf("Average reward: %d\n", mean(REWARDS));

plot(discount(REWARDS, ALPHA));
