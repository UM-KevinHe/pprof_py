# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pprof_py'
copyright = '2025, Kevin He'
author = 'Hevin He'

release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',  # Automatically generate documentation from docstrings
    'sphinx.ext.napoleon',  # Support for Google-style and NumPy-style docstrings
    'sphinx.ext.viewcode',  # Add links to the source code of documented objects
    'sphinx.ext.autosummary',  # Generate summary tables for modules and classes
    'sphinx_rtd_theme',  # ReadTheDocs theme for better styling
    'sphinx.ext.intersphinx',# Link to other projects' docs (e.g., Python, NumPy)
    'sphinx.ext.githubpages',# Support for GitHub Pages
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme' # Use the ReadTheDocs theme
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True # Or False if only using Google style
napoleon_include_init_with_doc = False
# ... other napoleon settings ...

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'nnumpy': ('https://numpy.org/doc/stable/', None),
}
