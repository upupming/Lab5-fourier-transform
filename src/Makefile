PY = python
all: wav_helper fourier_transform 2d_fourier_transform sin_fft boxcar_windowing_fft low_filter white_noise png_diff

wav_helper: wav_helper.py
	${PY} wav_helper.py > ../frames/wavs.txt

fourier_transform: fourier_transform.py
	${PY} fourier_transform.py

2d_fourier_transform: 2d_fourier_transform.py
	${PY} 2d_fourier_transform.py

sin_fft: sin_fft.py
	${PY} sin_fft.py

boxcar_windowing_fft: boxcar_windowing_fft.py
	${PY} boxcar_windowing_fft.py

low_filter: low_filter.py
	${PY} low_filter.py

white_noise: white_noise.py
	${PY} white_noise.py

png_diff: png_diff.py
	${PY} png_diff.py >> ../png_diff/png_diff.txt