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
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_theme_options = {
    "logo": {
        "text": "SALTRO",
    },
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    
    # Empty this list to remove all icons (including GitHub)
    "icon_links": [], 

    "show_nav_level": 2,
}
html_logo = "_static/rexlab.png"
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
]

breathe_projects = {
    "saltro": "./xml"
}
breathe_default_project = "saltro"
suppress_warnings = ["duplicate_declaration.cpp"]