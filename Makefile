spiral_generative.mp4 spiral_hybrid.mp4 spiral_discriminative.mp4: spiral.py
	pipenv run python3 spiral.py
	ffmpeg -y \
		-r 2 \
		-f image2 \
		-s 1920x1080 \
		-i plots/spiral_generative_%d.png \
		-vcodec libx264 \
		-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
		-crf 25 \
		-pix_fmt yuv420p \
		spiral_generative.mp4
	ffmpeg -y \
		-r 2 \
		-f image2 \
		-s 1920x1080 \
		-i plots/spiral_hybrid_%d.png \
		-vcodec libx264 \
		-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
		-crf 25 \
		-pix_fmt yuv420p \
		spiral_hybrid.mp4
	ffmpeg -y \
		-r 2 \
		-f image2 \
		-s 1920x1080 \
		-i plots/spiral_discriminative_%d.png \
		-vcodec libx264 \
		-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
		-crf 25 \
		-pix_fmt yuv420p \
		spiral_discriminative.mp4
