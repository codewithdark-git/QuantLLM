# Configuration file for the Sphinx documentation builder.

project = 'QuantLLM'
copyright = '2024, QuantLLM Team'
author = 'QuantLLM Team'
release = '2.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Support markdown
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
