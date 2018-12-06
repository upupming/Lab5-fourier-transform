import numpy
import matplotlib.pyplot as plt

M = 100
N = 1000
X = numpy.empty((M, N))

mean = 0
std = 2

for m in range(M):
    X[m] = numpy.random.normal(mean, std, size=N)

cross_correlation = numpy.empty(M)
for m in range(M):
    cross_correlation[m] = numpy.correlate(X[0], X[m])

plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Helvetica', 'Calibri']
plt.rcParams['font.size'] = 40
plt.rcParams['figure.figsize'] = 50, 10
plt.title('相关性')
plt.xlabel('白噪声信号')
plt.ylabel('相关性大小')
plt.plot(cross_correlation)
plt.savefig(f'../figures/white-noise-correlation.svg', bbox_inches='tight')
plt.close()