import os
import sys

import skala

# Ensure CONDA_PREFIX is set so CuPy can locate the CUDA toolkit
# when notebooks are executed during the Sphinx build.
os.environ.setdefault("CONDA_PREFIX", sys.prefix)

project = "Skala"
version = skala.__version__
author = "Microsoft Research, AI for Science"

extensions = [
    "myst_nb",
    "sphinx_book_theme",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.moderncmakedomain",
    "sphinxfortran.fortran_domain",
]

nb_execution_timeout = 300  # 5 minutes, set to -1 for no timeout
nb_execution_raise_on_error = True  # Fail the build on any notebook execution error
nb_merge_streams = True  # Merge multiple outputs from the same cell into one box

bibtex_bibfiles = [
    "_static/bib/gauxc.bib",
    "_static/bib/data.bib",
]

html_theme = "sphinx_book_theme"
html_title = project
html_logo = "_static/img/density.png"
html_favicon = "_static/img/density.png"
html_theme_options = {
    "repository_url": "https://github.com/microsoft/skala",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
}
master_doc = "index"

suppress_warnings = ["misc.highlighting_failure"]
exclude_patterns = ["_build", "jupyter_execute"]

# DOIs 403 automated requests from linkcheck, so we ignore the `doi.org` prefix
linkcheck_ignore = [
    r"^https://doi\.org/",
    # TODO: remove once arXiv v6 is posted.
    r"^https://arxiv\.org/abs/2506\.14492v6$",
]
