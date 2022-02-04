from distutils.core import setup

setup(
    name='SBbadger',
    packages=['SBbadger'],
    version='1.0.0',
    license='Apache',
    description='Synthetic biochemical reaction networks with definable degree distributions.',
    author='Michael Kochen',
    author_email='kochenma@uw.edu',
    url='https://github.com/sys-bio/SBbadger',
    download_url='https://github.com/sys-bio/SBbadger/archive/refs/tags/v1.0.0-beta.tar.gz',
    keywords=['Systems biology', 'Benchmark Models'],
    install_requires=[
        'numpy',
        'scipy',
        'antimony',
        'matplotlib',
        'pydot',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)
