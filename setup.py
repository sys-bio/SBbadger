from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='SBbadger',
    packages=['SBbadger'],
    version='1.0.0.6',
    license='Apache',
    description='Synthetic biochemical reaction networks with definable degree distributions.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Michael Kochen',
    author_email='kochenma@uw.edu',
    url='https://github.com/sys-bio/SBbadger',
    download_url='https://github.com/sys-bio/SBbadger/archive/refs/tags/v1.0.0.6.tar.gz',
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
