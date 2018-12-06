import wave
from matplotlib import pyplot

def get_frames(filename):
    """
    Return a list of numbers whose value are wave's amplitude.
    """
    wave_read_obj = wave.open(f'../wav/{filename}', 'rb')
    nframes = wave_read_obj.getnframes()
    frames = wave_read_obj.readframes(nframes)
    wave_read_obj.close()
    return list(frames)

def save_wav_to_svg(filename):
    """
    Convert .wav file to .svg file.
    """
    wave_read_obj = wave.open(f'../wav/{filename}', 'rb')
    
    print(f'========== begin {filename} info ==========')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wave_read_obj.getparams()
    frames = wave_read_obj.readframes(nframes)
    print('Number of audio channels:', nchannels)
    print('Sample width in bytes:', sampwidth)
    print('Sampling frequency:', framerate)
    print('Number of audio frames:', nframes)
    assert comptype == 'NONE'
    assert compname == 'not compressed'
    print(f'========== end {filename} info ============\n')

    pyplot.plot(list(frames))
    pyplot.title(filename, fontsize=80)

    # Save to image
    pyplot.savefig(f'../figures/{filename}.svg', bbox_inches='tight')
    pyplot.close()

    wave_read_obj.close()




    save_wav_to_svg('guitartune.wav')