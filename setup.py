#!/usr/bin/env python

import sys
import os
import re
from setuptools import setup, find_packages


def load_readme():
    with open('README.rst', 'r') as fd:
        return fd.read()


def load_requirements():
    """Parse requirements.txt"""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as fd:
        requirements = [line.rstrip() for line in fd]
    return requirements


sys.path.append("./tests")
package_name = 'causaltestdata'

init_path = os.path.join(os.path.dirname(__file__), package_name, '__init__.py')
with open(init_path) as f:
    version = re.search("__version__ = '([^']+)'", f.read()).group(1)

setup(name=package_name,
      version=version,
      description='A toolkit to generate pseudo dataset for causal inference.',
      long_description=load_readme(),
      author='Satoru Kobayashi',
      author_email='sat@nii.ac.jp',
      url='https://github.com/cpflat/causaltestdata/',
      install_requires=load_requirements(),
      classifiers=[
          'Development Status :: 4 - Beta',
          'Environment :: Console',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          "Intended Audience :: Developers",
          'License :: OSI Approved :: BSD License',
          "Operating System :: OS Independent",
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Software Development :: Libraries :: Python Modules'],
      license='The 3-Clause BSD License',

      packages=find_packages(),
      include_package_data=True,
      test_suite="tests"
      )
