#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

setup(
    author="HungCheng Chen",
    author_email="hcchen.nick@gmail.com",
    name="rewafer_multidefect",
    keywords="rewafer_multidefect",
    packages=find_packages(
        include=["rewafer_multidefect", "rewafer_multidefect.*"]
    ),
    test_suite="tests",
    license="MIT license",
    version="0.1.0",
)
