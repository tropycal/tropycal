# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAUTOGEN = sphinx-autogen
SPHINXPROJ    = tropycal
SOURCEDIR     = .
BUILDDIR      = _build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Clean from gallery
clean:
	rm -rf $(BUILDDIR)/* $(SOURCEDIR)/examples $(SOURCEDIR)/tutorials $(SOURCEDIR)/api/generated

.PHONY: help Makefile

# Makefile
%: Makefile
	@$(SPHINXAUTOGEN) -i -t $(SOURCEDIR)/_templates -o $(SOURCEDIR)/api/generated $(SOURCEDIR)/api/*.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)