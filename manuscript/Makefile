TARGET   = 0_main_icml2022
PROPOSALFILES = $(wildcard *.tex)
BIBFILES =  $(wildcard ./include/*.bib)
FIGURES  = 

.SUFFIXES: .plot .eps .fig

$(TARGET).pdf: $(TEXFILES) $(BIBFILES)
	pdflatex $(TARGET).tex
	bibtex $(TARGET)
	pdflatex $(TARGET).tex
	pdflatex $(TARGET).tex

all: $(TARGET).pdf

show:
	open $(TARGET).pdf

clean:
	rm -fv *.dvi *.aux *.log *~ *.bbl *.blg *.toc *.ps *.pdf *.out *.gz *.brf


distclean:: clean
	rm $(TARGET).ps

