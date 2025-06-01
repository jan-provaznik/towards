# 2023 - 2024 Jan Provaznik (provaznik@optics.upol.cz)
#

TARGETS = build/towards.pdf
DEPLIST = towards.bib 

.PHONY:
all: ${TARGETS}

.PHONY:
clean: 
	rm -rf build

build: 
	mkdir -p build

build/%.pdf: %.tex ${DEPLIST} | build
	latexmk --pdflatex --halt-on-error --output-directory=build $<

build/OE.tgz: ${TARGETS} ${DEPLIST}
	rm -rf build/OE
	mkdir -p build/OE
	cp -r import build/OE
	cp -r styles build/OE
	cp jabbrv-ltwa-all.ldf jabbrv-ltwa-en.ldf jabbrv.sty build/OE
	cp optica-article.cls opticajnl.bst opticajnlnt.bst  build/OE
	cp towards.{tex,bib} build/towards.bbl build/OE
	tar -zcf $@ -C build/OE -- .
	rm -rf build/OE

