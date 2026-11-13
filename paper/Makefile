LATEXMK ?= latexmk
PYTHON ?= python3
CONTROLLER_FIGURE := figures/controller_diagrams.pdf
TRAINING_FIGURE := figures/dreamer_training_curves.pdf

.PHONY: all figures diagrams clean

all: main.pdf

main.pdf: main.tex references.bib $(CONTROLLER_FIGURE) $(TRAINING_FIGURE)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

# The checked-in PDFs allow paper builds without private training records.
figures: diagrams

diagrams:
	$(PYTHON) draw_controller_diagrams.py

clean:
	$(LATEXMK) -C
