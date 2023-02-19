project = "vulkpy"
author = "Hiroyuki Yamada"
copyright = "2023, Hiroyuki Yamada"

extensions = [
    'sphinx.ext.napoleon',
    "sphinx_automodapi.automodapi",
    'sphinx_automodapi.smart_resolver',
    'myst_parser'
]

html_title = "vulkpy"
html_theme = "furo"
html_logo = ""
html_favicon = ""
html_show_sourcelink = False

html_css_files = [
      "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
      "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
      "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]

html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/ymd-h/vulkpy",
            "html": "",
            "class": "fa-brands fa-github fa-2x",
        },
    ],
}

napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

numpydoc_show_class_members=False

autodoc_default_options = {
    'member-order': 'bysource',
    'class-doc-from':'both',
    'exclude-members': '__dict__, __weakref__, __module__, __new__, __reduce__, __setstate__',
    'inherited-members': True
}
