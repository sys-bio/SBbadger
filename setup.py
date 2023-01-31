from setuptools import setup
import os.path
import codecs

with open("README.md", "r") as fh:
    long_description = fh.read()

# The following two methods were copied from
# https://packaging.python.org/guides/single-sourcing-package-version/#single-sourcing-the-version


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            print(line)
            delim = '"' if '"' in line else "'"
            print('delim = ', delim)
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name='SBbadger',
    packages=['SBbadger'],
    version=get_version("SBbadger/_version.py"),
    license='Apache',
    description='Synthetic biochemical reaction networks with definable degree distributions.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Michael Kochen',
    author_email='kochenma@uw.edu',
    url='https://github.com/sys-bio/SBbadger',
    download_url='https://github.com/sys-bio/SBbadger/archive/refs/tags/v1.2.7.tar.gz',
    keywords=['Systems biology', 'Benchmark Models'],
    install_requires=[
        'numpy',
        'scipy',
        'antimony',
        'matplotlib',
        'pydot',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
