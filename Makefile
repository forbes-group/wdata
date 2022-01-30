PANDOC_FLAGS = --toc --standalone

all: README-PYPI.html README-test.html

README-PYPI.html: README.md
	python -m readme_renderer $< -o $@	

README-test.html: README.md
	pandoc -o $@ $< 

test:
	nox

%.html: %.md
	pandoc $(PANDOC_FLAGS) $< -o $@  && open -g -a Safari $@
	fswatch -e ".*\.html" -o . | while read num ; do pandoc $(PANDOC_FLAGS) $< -o $@ && open -g -a Safari $@; done

.PHONY: all clean test

clean:
	-rm -rf .nox
	-rm README*.html
	-rm -rf fil-result
	-rm -rf dist
	-rm -f coverage.xml
	-find . -type d -name "__pycache__" -exec rm -rf {} \;
	-find . -type f -name "*.pyc" -delete
