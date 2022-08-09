import sphinx_material

project = "Rexify"
html_title = "Rexify"

html_theme = 'sphinx_material'

extensions = [
    "sphinx.ext.autodoc",
    # Creates .nojekyll config
    "sphinx.ext.githubpages",
    # Converts markdown to rst
    "m2r2",
    "sphinx.ext.napoleon",
    "sphinxcontrib.apidoc",  # automatically generate API docs, see https://github.com/rtfd/readthedocs.org/issues/1139
    "sphinx_search.extension",
]
source_suffix = [".rst", ".md"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False

if html_theme == "sphinx_material":
    html_theme_options = {
        "color_primary": "teal",
        "color_accent": "light-blue",
        "repo_url": "https://github.com/joseprsm/rexify",
        "repo_name": "Rexify",
        "globaltoc_depth": 2,
        "globaltoc_collapse": False,
        "globaltoc_includehidden": False,
        "repo_type": "github"
    }

    extensions.append("sphinx_material")
    html_theme_path = sphinx_material.html_theme_path()
    html_context = sphinx_material.get_html_context()

