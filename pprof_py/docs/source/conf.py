```python
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))  # Points to the project root directory

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------

project = 'pprof_py'
copyright = '2025, Kevin He'
author = 'Kevin He'

release = '0.2.0'
version = '0.2'


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'sphinxcontrib.bibtex',
    'sphinx.ext.mathjax',
    'myst_parser',
    'matplotlib.sphinxext.plot_directive',
]


# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'colon_fence',
    'deflist',
]

myst_heading_anchors = 3


# -- Source file configuration -----------------------------------------------

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []


# -- Bibliography ------------------------------------------------------------

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'author_year'


# -- Autosummary -------------------------------------------------------------

autosummary_generate = True


# -- HTML output --------------------------------------------------------------

html_theme = 'shibuya'

html_static_path = ['_static']

html_title = 'pprof_py'

html_theme_options = {
    'github_url': 'https://github.com/UM-KevinHe/pprof_py',
}


# -- Napoleon settings --------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True


# -- Autodoc settings ---------------------------------------------------------

autodoc_member_order = 'bysource'


# -- Intersphinx --------------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}
