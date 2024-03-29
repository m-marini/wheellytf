tx = ty = linspace (-3, 3, 65)';
[xx, yy] = meshgrid (tx, ty);
DIST = sqrt(xx .* xx + yy .* yy);
ANG = atan2(yy, xx) * 180 / pi;
tz = fuzzy_stuck(DIST, ANG);
#mesh (tx, ty, tz);
surf (tx, ty, tz);
xlabel ("tx");
ylabel ("ty");
zlabel ("tz");
