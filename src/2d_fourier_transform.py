import numpy
import bmp_helper as bh
import matplotlib.pyplot as plt

def plot_amplitude_spectrum(frames):
    A = numpy.fft.fft2(frames)
    A = numpy.fft.fftshift(A)
    A = numpy.abs(A)
    A = numpy.log(A)
    
    plt.imshow(A)

def plot_phase_spectrum(frames):
    A = numpy.fft.fft2(frames)
    phase = numpy.angle(A)
    
    plt.imshow(phase)

def plot_inverse_by_amplitude_only(frames):
    A = numpy.fft.fft2(frames)
    A = numpy.fft.fftshift(A)
    A = numpy.abs(A)
    
    inverse = numpy.fft.ifft2(A)
    plt.imshow(numpy.abs(inverse))

def plot_inverse_by_phase_only(frames):
    A = numpy.fft.fft2(frames)
    phase = numpy.angle(A)
    
    inverse = numpy.fft.ifft2(phase)
    plt.imshow(numpy.abs(inverse))

def plot_inverse(frames):
    A = numpy.fft.fft2(frames)

    inverse = numpy.fft.ifft2(A)
    plt.imshow(numpy.abs(inverse))

def save_spectrum(filename):
    frames = bh.get_array(filename)
    
    # Amplitude spectrum
    plt.title(f'{filename} - 幅度谱')
    plot_amplitude_spectrum(frames)
    plt.savefig(f'../figures/{filename}-amplitude.svg', bbox_inches='tight')
    plt.close()

    # Phase spectrum
    plt.title(f'{filename} - 相位谱')
    plot_phase_spectrum(frames)
    plt.savefig(f'../figures/{filename}-phase.svg', bbox_inches='tight')
    plt.close()

    # Inverse using amplitude only
    plt.title(f'{filename} - 只对幅度进行逆傅里叶变换')
    plot_inverse_by_amplitude_only(frames)
    plt.savefig(f'../figures/{filename}-inverse-by-amplitude.svg', bbox_inches='tight')
    plt.close()

    # Inverse using phase only
    plt.title(f'{filename} - 只对相位进行逆傅里叶变换')
    plot_inverse_by_phase_only(frames)
    plt.savefig(f'../figures/{filename}-inverse-by-phase.svg', bbox_inches='tight')
    plt.close()

    # Inverse using all information
    plt.title(f'{filename} - 逆傅里叶变换')
    plot_inverse(frames)
    plt.savefig(f'../figures/{filename}-inverse.svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Helvetica', 'Calibri']
    save_spectrum('Girl.bmp')
    save_spectrum('Sonic.bmp')
    save_spectrum('LENA.BMP')