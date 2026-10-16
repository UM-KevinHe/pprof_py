import os
import sys
sys.path.insert(0, os.path.abspath('../../..'))  # Repository root (one level above pprof_py/)
sys.path.insert(0, os.path.abspath('_ext'))  # local extensions (the presentation gallery)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pprof_py'
copyright = '2025, Kevin He'
author = 'Kevin He'

release = '0.7.0'
version = '0.7'

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
    'pprof_gallery',
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

autosummary_generate = False  # no page uses autosummary directives

# templates_path = ['_templates']  # No custom templates yet
exclude_patterns = ['presentation/_gallery_items.md', 'presentation/_figures/*']  # generated includes

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'shibuya'
html_static_path = ['_static']
html_css_files = ['custom.css']  # reading-experience adjustments on top of the theme
html_js_files = ['fontsize.js']  # A- / A+ text-size control next to the colour-mode toggle
html_theme_options = {
    'globaltoc_expand_depth': 1,
    'toctree_titles_only': True,
    'color_mode': 'auto',          # follow the system, with the light/dark toggle in the header
    'nav_links': [
        # One entry per documentation section, in the order of the sidebar.
        {'title': 'Guides', 'children': [
            {'title': 'Introduction', 'url': 'introduction'},
            {'title': 'Model architecture', 'url': 'model_architecture'},
            {'title': 'Core concepts', 'url': 'direct_vs_indirect_standardization'},
            {'title': 'Linear models', 'url': 'linear/00_start_here'},
            {'title': 'Logistic models', 'url': 'logistic/00_start_here'},
            {'title': 'Survival analysis', 'url': 'survival/00_start_here'},
            {'title': 'Presentation', 'url': 'presentation/index'},
            # TODO: switch to 'validation/index' once the validation branch is
            # merged; the page does not exist here yet, and nav_links are not
            # checked by the build, so a missing target gives a dead link
            # rather than a warning.
            {'title': 'Validation overview', 'url': 'survival/VALIDATION_REPORT'},
            {'title': 'Reference', 'url': 'reference/linear_models'},
            {'title': 'API reference', 'url': 'api'},
        ]},
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
