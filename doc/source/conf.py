# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys

sys.path.insert(0, '../')

project = 'Image Alignment'
copyright = '2024, SMCP-PFNC'
author = 'SMCP-PFNC'
release = '2024.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx.ext.autosummary',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.mathjax',
    'nbsphinx',
    'nbsphinx_link',
    'sphinx_mdinclude',
]
autodoc_mock_imports = ['csbdeep', 'stardist', 'tensorflow']

# math_number_all = False

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = True

# Use to also document __init__ method of class.
autoclass_content = "both"

# This value selects if automatically documented members are sorted alphabetical
# (value 'alphabetical'), by member type (value 'groupwise') or by source order
# (value 'bysource'). The default is alphabetical.
# Note that for source order, the module must be a Python module with the source
# code available.
autodoc_member_order = "bysource"

templates_path = ['_templates']
exclude_patterns = []
# include_patterns = ['index.rst', ...]
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = r'_static/logo.png'

pygments_style = 'sphinx'
htmlhelp_basename = 'images_alignment'
pngmath_use_preview = True
pngmath_dvipng_args = ['-gamma', '1.5', '-D', '96', '-bg', 'Transparent']
todo_include_todos = True
nbsphinx_allow_errors = True


def run_apidoc(_):
    # https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html
    # https://www.sphinx-doc.org/es/1.2/ext/autodoc.html
    import os
    os.environ['SPHINX_APIDOC_OPTIONS'] = 'members,' \
                                          'private-members,' \
                                          'no-undoc-members,' \
                                          'show-inheritance,' \
                                          'ignore-module-all'

    from sphinx.ext.apidoc import main

    cur_dir = os.path.normpath(os.path.dirname(__file__))
    output_path = os.path.join(cur_dir, 'api')
    modules = os.path.normpath(os.path.join(cur_dir, "../../images_alignment"))
    exclude_pattern = []
    main(['-e', '-f', '-P', '-o', output_path, modules, *exclude_pattern])


def setup(app):
    app.connect('builder-inited', run_apidoc)
