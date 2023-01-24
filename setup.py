#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="ignite_template",
    version="0.0.1",
    description="Template for training pipline on pytorch-ignite.",
    author="yuva",
    author_email="",
    url=
    "https://github.com/user/project",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["hydra-core"],
    packages=find_packages(),
)
