SOC = 0:0.001:1;
matrix(:,1) = SOC;
matrix(:,2) = V_avgC20';
csvwrite('Voc.dat',matrix);