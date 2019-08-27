# This Makefile is based on https://github.com/mattpitkin/psrqpy/blob/master/paper/Makefile
default: pdf

MDFILE=paper.md
BIBFILE=paper.bib
LATEXFILE=paper.tex

# some variables for the paper
REPOSITORY=https://github.com/duetosymmetry/qnm
ARCHIVE=https://doi.org/10.5281/zenodo.2593978
REVIEW=https://github.com/openjournals/joss-reviews/issues/1683
ARCHIVEDOI=https://doi.org/10.5281/zenodo.2593978
FORMATTEDDOI=10.21105/joss.01683
YEAR=2019
TITLE="qnm: A Python package for calculating Kerr quasinormal modes, separation constants, and spherical-spheroidal mixing coefficients"
AUTHOR="Stein"
ISSUE=40
VOLUME=4
PAGE=1683
JOURNALNAME="Journal of Open Source Software"
PUBLISHED="XXX XXX 2019"
SUBMITTED="21 August 2019"
PAPERURL=https://www.theoj.org/joss-papers/joss.XXXXX/10.21105.joss.XXXXX.pdf

latex.template:
	# get JOSS template
	curl -O https://raw.githubusercontent.com/openjournals/whedon/master/resources/latex.template

joss-logo.png:
	# get JOSS logo
	curl -O https://raw.githubusercontent.com/openjournals/whedon/master/resources/joss-logo.png

latex: $(MDFILE) $(BIBFILE) latex.template joss-logo.png
	# use pandoc and pandoc-citeproc to create LaTeX source
	pandoc --filter pandoc-citeproc --bibliography=$(BIBFILE) -V repository="$(REPOSITORY)" -V journal_name=$(JOURNALNAME) -V paper_url="$(PAPERURL)" -V archive_doi="$(ARCHIVEDOI)" -V review_issue_url="$(REVIEW)" -V formatted_doi="$(FORMATTEDDOI)" -V logo_path="joss-logo.png" -V year="$(YEAR)" -V footnote_paper_title=$(TITLE) -V citation_author=$(AUTHOR) -V issue="$(ISSUE)" -V volume="$(VOLUME)" -V page="$(PAGE)" -V submitted=$(SUBMITTED) -V published=$(PUBLISHED) -V graphics="true" -V papersize=a4paper -V geometry:margin=1in -t latex -s $(MDFILE) -o $(LATEXFILE) --template latex.template

pdf: latex
	pdflatex paper.tex
	pdflatex paper.tex

clean:
	rm paper.bcf paper.aux paper.log paper.out paper.run.xml

clean_downloads:
	rm latex.template joss-logo.png
