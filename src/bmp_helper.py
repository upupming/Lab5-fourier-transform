from matplotlib import pyplot

def get_array(filename):
    """
    Read an image from a bmp file into an array.
    """
    return pyplot.imread(f'../bmp/{filename}', 'bmp')