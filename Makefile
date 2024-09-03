# 2023 - 2024 Jan Provaznik (provaznik@optics.upol.cz)
#

TARGETS = build/towards.pdf
DEPLIST = towards.bib protools.sty 

.PHONY:
all: ${TARGETS}

.PHONY:
clean: 
	rm -rf build

# Master builder (rules)
#

build/%.pdf: %.tex ${DEPLIST}
	latexmk --xelatex --halt-on-error --output-directory=build $<

