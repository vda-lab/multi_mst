#!/usr/bin/env python3

"""
Determine the name of the wheel and export to GitHub output named WHEEL_NAME.

To run:
    $ python3 -m build --sdist
    $ ./ci/determine_name.py
"""
import os
from pathlib import Path
import sys


paths = [p.name for p in Path("dist").glob("*.whl")]
if len(paths) != 1:
    sys.exit(f"Only a single wheel is supported, but found: {paths}")

print(paths[0])
with open(os.environ["GITHUB_OUTPUT"], "a") as f:
    f.write(f"WHEEL_NAME={paths[0]}\n")