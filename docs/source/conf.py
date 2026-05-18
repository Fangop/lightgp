# Copyright (c) 2026 Yu-Hsueh Fang. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root.

"""Sphinx configuration for the LightGP documentation."""

from __future__ import annotations

import os
import sys

# Make the locally-built Python package importable for autodoc, even though we
# do not actually run autodoc-from-import in the default build (the C++
# extension is platform-specific). The path is set up defensively in case a
# future page does want to introspect the package.
DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(DOCS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

# -- Project information -----------------------------------------------------
project = "LightGP"
author = "YuHsueh Fang"
copyright = "2026, YuHsueh Fang"
release = "0.1.1"
version = "0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",        # Google / NumPy docstring style
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",          # copy-button on code blocks
    "sphinx_design",              # cards, grids, tabs
    "myst_parser",                # Markdown support
    "nbsphinx",                   # render Jupyter notebooks
]

# nbsphinx: the tutorial notebooks are pre-executed by
# docs/build_tutorial_notebooks.py and committed with outputs. Build is fast
# and reproducible — Sphinx never runs Python code at HTML-build time.
nbsphinx_execute = "never"
nbsphinx_allow_errors = False

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# MyST
myst_enable_extensions = ["colon_fence", "deflist"]

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "prev_next_buttons_location": "bottom",
}
html_title = f"{project} {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Copy-button: skip the prompts so users don't paste them.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
