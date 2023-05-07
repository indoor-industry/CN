import numpy as np
import matplotlib.pyplot as plt

M=N=10
Tc = [0.5, 1.4473684210526314, 2.394736842105263, 2.8684210526315788, 3.81578947368421, 4.526315789473684, 4.526315789473684]
lenth_p_sweep = np.arange(1/(N*M), 7.5/(N*M), 1/(M*N))
print(lenth_p_sweep)

plt.scatter(lenth_p_sweep, Tc)
plt.title('ER Critical temperature')
plt.xlabel('p')
plt.ylabel('Tc')
plt.show()