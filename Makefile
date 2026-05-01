.PHONY: all assets slides clean

all: slides

assets:
	python3 scripts/generate_assets.py

slides: assets
	cd tex && latexmk -xelatex -interaction=nonstopmode -file-line-error slides.tex

clean:
	cd tex && latexmk -c slides.tex
	rm -rf assets/frames_* src/__pycache__ scripts/__pycache__
