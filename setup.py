#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="ignite_template",
    version="0.0.12",
    description="Template for training pipline on pytorch-ignite.",
    author="yuva",
    author_email="",
    install_requires=["hydra-core", "loguru"],
    packages=find_packages(),
)
