import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))  # Repository root (one level above pprof_py/)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pprof_py'
copyright = '2025, Kevin He'
author = 'Kevin He'

release = '0.4.1'
version = '0.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

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

# -- MyST (Markdown) configuration -------------------------------------------
myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'colon_fence',
    'deflist',
]
myst_heading_anchors = 2

# Supported source suffixes
source_suffix = {
    '.md': 'markdown',
}

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'author_year'

autosummary_generate = True

# templates_path = ['_templates']  # No custom templates yet
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
html_static_path = ['_static']
html_theme_options = {
    'globaltoc_expand_depth': 1,
    'toctree_titles_only': True,
    'nav_links': [
        {'title': 'GitHub', 'url': 'https://github.com/UM-KevinHe/pprof_py'},
    ],
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

autodoc_member_order = 'bysource'
autodoc_default_options = {
    'exclude-members': '__weakref__, get_params, set_params',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}
