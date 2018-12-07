from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
import numpy

def png_to_array(filename):
    return plt.imread(f'../avi/{filename}', 'png')

frame_prefix = 'video-frame'
frame_suffix = '.png'

frame1 = '00001'
frame150 = '00150'

array1 = png_to_array(frame_prefix + frame1 + frame_suffix)
(rows, cols, channels) = array1.shape
array1 = array1.reshape(rows*cols, channels)

array150 = png_to_array(frame_prefix + frame150 + frame_suffix).reshape(rows*cols, channels)

print('Running kmeans on 1st image...')
(codebook, distortion) = kmeans(array1, 2)
print('kmeans result:')
print((codebook, distortion))

print('Masking 150th image using the calculated centroids...')
masked_gray_array150 = numpy.empty((rows, cols))
for i in range(rows):
    for j in range(cols):
        value = array150[i * cols + j]
        if(numpy.linalg.norm(value - codebook[0]) < numpy.linalg.norm(value - codebook[1])):
            masked_gray_array150[i][j] = 0
        else:
            masked_gray_array150[i][j] = 255

plt.imshow(masked_gray_array150)
plt.savefig(f'../figures/png-diff.svg', bbox_inches='tight')