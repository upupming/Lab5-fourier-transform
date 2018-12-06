import numpy
import wav_helper as wh
import matplotlib.pyplot as plt

def plot_amplitude_spectrum(frames):
    # All element minus mean, this will avoid the zero-frequency component(sum of signal) increase with number of samples, thus much more larger than else frequencies finally
    frames = frames - numpy.mean(frames)
    A = numpy.fft.rfft(frames)
    freq = numpy.fft.rfftfreq(frames.shape[-1])
    plt.plot(freq, numpy.abs(A))

def plot_phase_spectrum(frames):
    # frames = frames - numpy.mean(frames)
    A = numpy.fft.rfft(frames)
    phase = numpy.angle(A)
    freq = numpy.fft.rfftfreq(frames.shape[-1])
    plt.plot(freq, phase)

def save_spectrum(filename, frames=None):
    if frames.all() == None:
        frames = numpy.array(wh.get_frames(filename))
    
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

if __name__ == "__main__":
    save_spectrum('crane_bump.wav')
    save_spectrum('engine.wav')
    save_spectrum('guitartune.wav')