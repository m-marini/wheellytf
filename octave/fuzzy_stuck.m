function Z = fuzzy_stuck(DISTANCES, ANGLES, D0=0.1, D1=0.3, D2=0.7, D3=2, RANGE=90)
  X0 = DISTANCES - D0;
  X1 = X0 / (D1 - D0);
  X2 = DISTANCES - D3;
  X3 = -X2 / (D3 - D2);
  X4 = min(X1, X3);
  
  X6 = 1- abs(ANGLES) / RANGE;
  Z = min(X4, X6);
  Z = min(max(Z, 0), 1);
end
