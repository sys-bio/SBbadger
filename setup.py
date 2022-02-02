from distutils.core import setup

setup(
    name='SBbadger',  # How you named your package folder (MyLib)
    packages=['SBbadger'],  # Chose the same as "name"
    version='0.1.10.1',
    license='Apache',
    description='Synthetic biochemical reaction networks with definable degree distributions.',  # Give a short description about your library
    author='YOUR NAME',  # Type in your name
    author_email='kochenma@uw.edu',  # Type in your E-Mail
    url='https://github.com/sys-bio/SBbadger',  # Provide either the link to your github or to your website
    download_url='https://github.com/sys-bio/SBbadger/archive/refs/tags/v0.1.10.1-alpha.tar.gz',  # I explain this later on
    keywords=['Systems biology', 'Benchmark Models'],  # Keywords that define your package best
    install_requires=[  # I get to this in a second
        'numpy',
        'scipy',
        'antimony',
        'matplotlib',
        'pydot',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache Software License',  # Again, pick a license
        'Programming Language :: Python :: 3',  # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.6',
    ],
)
