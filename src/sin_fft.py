import fourier_transform as ft
import numpy
import matplotlib.pyplot as plt

size = 1000
x = numpy.arange(size)
frames = numpy.zeros(size)
for w in range(1, 20, 2):
    frames += numpy.sin(w * x)

plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Helvetica', 'Calibri']
plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = 50, 10

plt.plot(x, frames)
plt.title('正弦合成信号')
plt.savefig(f'../figures/sum-of-sin.svg', bbox_inches='tight')
plt.close()

ft.save_spectrum("sum-of-sin", frames)