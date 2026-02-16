# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SALTRO'
copyright = '2026, Niclas Scheuer, Patrick McKeen'
author = 'Niclas Scheuer, Patrick McKeen'
release = '0.0.0'

extensions = [
    "breathe",
    "sphinx.ext.mathjax",
    "sphinx_rtd_dark_mode"
]
default_dark_mode = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_theme_options = {
    "style_external_links": True,
}

breathe_projects = {
    "saltro": "./xml"
}
breathe_default_project = "saltro"