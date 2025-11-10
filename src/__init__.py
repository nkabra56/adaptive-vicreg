"""
Make the top-level 'src' directory a Python package.

Why:
- Our CLI scripts import using `from src.vicreg_tf import ...`.
- Adding this file ensures Python treats 'src' as a package so those imports work
  without fiddling with PYTHONPATH.

Author: Nishant Kabra
Date: 11/8/2025
"""
# No runtime code needed here.
