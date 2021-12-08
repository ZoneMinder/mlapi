#!/usr/bin/env python3

import os
import re
import codecs

from setuptools import setup
with open('requirements.txt') as f:
    requirements = f.readlines()
# Package meta-data.
NAME = 'neo-mlapi'
DESCRIPTION = 'Machine Learning API using the flask framework'
URL = 'https://github.com/baudneo/mlapi'
AUTHOR_EMAIL = 'baudneo@protonmail.com'
AUTHOR = 'Pliable Pixels forked by baudneo'
LICENSE = 'GPL'
INSTALL_REQUIRES = requirements

here = os.path.abspath(os.path.dirname(__file__))
# read the contents of your README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        data = fp.read()
    return data


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name=NAME,
      python_requires='>=3.6',
      version=find_version('mlapi.py'),
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      url=URL,
      license=LICENSE,
      install_requires=INSTALL_REQUIRES,
      py_modules=[
          'neo-mlapi',
      ]
      )
