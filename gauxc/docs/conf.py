import skala

project = "Skala GauXC integration"
version = skala.__version__
author = "Microsoft Research, AI for Science"

extensions = [
    "sphinx_book_theme",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.moderncmakedomain",
    "sphinxfortran.fortran_domain",
]

bibtex_bibfiles = ["gauxc.bib"]

html_theme = "sphinx_book_theme"
html_title = project
html_theme_options = {
    "repository_url": "https://github.com/microsoft/skala",
    "repository_branch": "main",
    "path_to_docs": "gauxc/docs",
    "use_repository_button": True,
}
master_doc = "index"

exclude_patterns = ["_build"]

linkcheck_ignore = [r"^https://doi\.org/"]
