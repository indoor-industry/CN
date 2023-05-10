
independent = readtable('corr_r 20.csv')
dependent = readtable('r.csv')

x = table2array(dependent)
y = table2array(independent)

ft = fittype('a*exp(-b*x)+ c','dependent',{'y'},'independent',{'x'},'coefficients',{'a','b','c'});
curve = fit(x, y, ft, 'StartPoint', [0,0,0])

hold on
scatter(x, y)

plot(curve,'m')
legend('Data','n=2')

hold off
