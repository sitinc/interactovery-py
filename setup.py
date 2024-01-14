from setuptools import setup, find_packages

setup(
    # Basic package information:
    name="interactovery",
    version="0.0.6",
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    # Metadata for PyPI:
    author="Justin Randall",
    author_email="justin@sitinc.net",
    description="Python module for interactional discovery.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sitinc/interactovery-py",
    license="MIT",
    keywords="nlp nlu transcripts customer-journeys journeys",

    # Lock down Python version.
    python_requires='>=3.7.1, <3.12',

    # List of dependencies:
    install_requires=[
        "sentence-transformers",
        "hdbscan",
        "scikit-learn",
        "umap-learn",
        "pandas",
        "matplotlib",
        "seaborn",
        "numpy",
        "openai",
        "python-dotenv",
        "requests",
        "spacy"
    ],

    # Additional classifiers can help users find your project. For a full list,
    # see: https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 3 - Alpha",  # Change as appropriate
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],

    # This is a dictionary of additional data files that you want to include in your package.
    # The key is the package name, and the value is a list of relative directory names that
    # should be included in the package.
    package_data={
        # Include any '*.txt' and '*.rst' files found in the 'interactovery' package
        "interactovery": ["*.txt", "*.rst"]
    },

    # Although 'package_data' is the preferred approach, in some cases you may need to
    # place data files outside of your packages. See:
    # http://docs.python.org/distutils/setupscript.html#installing-additional-files
    data_files=[('my_data', ['data/data_file'])],

    # This is a string or list of strings specifying what other distributions need to be present
    # for this one to be installed or used. This can be used to specify dependencies that are not
    # in the Python standard library or available on PyPI.
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
)
