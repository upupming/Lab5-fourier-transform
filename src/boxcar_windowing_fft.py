import numpy
import matplotlib.pylab as plt
import fourier_transform as ft

######### Sampling signal points ################

# Sampling frequency, in Hz
f_s = 100
number_of_points = 512

start = 0
stop = (number_of_points - 1)/f_s
t = numpy.linspace(start, stop, number_of_points)

f_1 = 20
f_2 = 30
f_3 = 40
x = numpy.sin(2 * numpy.pi * 20 * t) \
    + numpy.sin(2 * numpy.pi * f_2 * t) \
    + numpy.sin(2 * numpy.pi * f_3 * t)
x_1 = x[0:128]

######### FFT using rectangular window ##############

plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Helvetica', 'Calibri']
plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = 50, 10

plt.plot(x_1)
plt.title('128 个样本信号')
plt.savefig(f'../figures/boxcar-128.svg', bbox_inches='tight')
plt.close()

ft.save_spectrum("boxcar-128", x_1)


plt.plot(x)
plt.title('512 个样本信号')
plt.savefig(f'../figures/boxcar-512.svg', bbox_inches='tight')
plt.close()

ft.save_spectrum("boxcar-512", x)