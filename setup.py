#!/usr/bin/env python
import builtins
import os
import pathlib

from setuptools import find_packages, setup

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
PATH_ROOT = pathlib.Path(__file__).parent
builtins.__COVID_SETUP__ = True


def load_requirements(
    path_dir=PATH_ROOT, file_name="requirements.txt", comment_char="#"
):  # noqa: D103
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http"):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


def load_long_description():  # noqa: D103
    text = open(PATH_ROOT / "README.md", encoding="utf-8").read()
    # SVG images are not readable on PyPI, so replace them  with PNG
    text = text.replace(".svg", ".png")
    return text


setup(
    name="covid",
    version="0.0.1",
    description="Color-Doppler intracardiac Vector flow Imaging using physics-constrained Deep learning",
    author="Hang Jung Ling",
    author_email="hang-jung.ling@insa-lyon.fr",
    url="https://github.com/HangJung97/CoVID",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=load_requirements(),
    packages=find_packages(),
    long_description=load_long_description(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    setup_requires=[],
)
