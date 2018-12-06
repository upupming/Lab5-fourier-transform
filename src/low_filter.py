import numpy
import matplotlib.pyplot as plt

Omega = numpy.linspace(0, 2 * numpy.pi, 1000)

# See equation (22)
a = [1, -1.76, 1.1829, -0.2781]
b = [0.0181, 0.0543, 0.0543, 0.0181]

# Calculate H(\Omega), see euqation (20)
H = \
    b[0] + b[1] * numpy.exp(-1j * Omega) + b[2] * numpy.exp(-2j * Omega) + b[3] * numpy.exp(-3j * Omega) / \
    a[0] + a[1] * numpy.exp(-1j * Omega) + a[2] * numpy.exp(-2j * Omega) + a[3] * numpy.exp(-3j * Omega)

def plot_amplitude_spectrum(frames):
    # All element minus mean, this will avoid the zero-frequency component(sum of signal) increase with number of samples, thus much more larger than else frequencies finally
    frames = frames - numpy.mean(frames)
    A = frames
    freq = numpy.fft.fftshift(numpy.fft.fftfreq(frames.shape[-1]))
    plt.plot(freq, numpy.abs(A))

def plot_phase_spectrum(frames):
    # frames = frames - numpy.mean(frames)
    A = frames
    phase = numpy.angle(A)
    freq = numpy.fft.fftshift(numpy.fft.fftfreq(frames.shape[-1]))
    plt.plot(freq, phase)

def save_spectrum(filename, frames):
    
    # Amplitude spectrum
    plt.title(f'{filename} - 幅度谱', fontsize=40)
    plt.xlabel('频率', fontsize=40)
    plt.ylabel('幅度', fontsize=40)
    plot_amplitude_spectrum(frames)
    plt.savefig(f'../figures/{filename}-amplitude.svg', bbox_inches='tight')
    plt.close()

    # Phase spectrum
    plt.title(f'{filename} - 相位谱', fontsize=40)
    plt.xlabel('频率', fontsize=40)
    plt.ylabel('相位', fontsize=40)
    plot_phase_spectrum(frames)
    plt.savefig(f'../figures/{filename}-phase.svg', bbox_inches='tight')
    plt.close()

plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Helvetica', 'Calibri']
plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = 50, 10
save_spectrum('low-filter', H)