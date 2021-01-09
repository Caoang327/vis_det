#!/usr/bin/env python
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vis_det",
    version="0.0.0",
    author="Ang Cao",
    description="code of Visualizing and Understanding Object Detector",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    license="MIT",
    zip_safe=True,
)
