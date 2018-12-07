SOURCE_DIR = ./src

all:
	cd $(SOURCE_DIR) && make all

%:
	cd $(SOURCE_DIR) && make $@

clean:
	rm figures/* frames/*

avi2png:
	ffmpeg -i "./avi/visiontraffic.avi" -f image2 "./avi/video-frame%05d.png"