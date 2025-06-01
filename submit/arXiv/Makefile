# 2023 - 2024 Jan Provaznik (provaznik@optics.upol.cz)
#

TARGETS = build/towards.pdf
DEPLIST = towards.bib arxtools.sty 

.PHONY:
all: ${TARGETS}

.PHONY:
clean: 
	rm -rf build

build: 
	mkdir -p build

build/%.pdf: %.tex ${DEPLIST} | build
	latexmk --pdflatex --halt-on-error --output-directory=build $<
towards-rl.tex: towards.tex.original towards.tex 
	latexdiff -p redhead.tex $^ > $@

build/arXiv.tgz: ${TARGETS} ${DEPLIST}
	rm -rf build/arXiv
	mkdir -p build/arXiv/protools
	cp -r protools/icons build/arXiv/protools
	cp -r import build/arXiv
	cp arxtools.sty towards.{tex,bib} build/towards.bbl build/arXiv
	tar -zcf $@ -C build/arXiv -- .
	rm -rf build/arXiv

