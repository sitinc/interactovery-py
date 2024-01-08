# interactovery-py
Python module for explore interactional and customer journey data.  A little bit of natural language processing (NLP), 
machine leaning (ML), and large language models (LLMs) to help you better understand what people are saying or writing.

## Installation

This project requires Python 3.x. Below are the instructions to set up the environment for this project.

### Installing interactovery module

You can install the interactovery module frmo PyPI:

```bash
pip install interactovery
```

### Installing spaCy Model

After installing the required packages, you need to download the spaCy English model "en_core_web_lg". Run the following command:

```bash
python -m spacy download en_core_web_lg
```

### Additional Requirement for Windows Users
For Windows users, to install hdbscan, you may need to install the build tools from Visual Studio. This is because hdbscan requires C++ compilation which is not natively supported in Windows Python environments.

You can download the Visual Studio Build Tools from [this link](https://visualstudio.microsoft.com/downloads/). Follow the instructions to install the necessary components.


## Usage

There is [a GitHub project](https://github.com/sitinc/journey-discovery-getting-started) with [various Jupyter notebooks](https://github.com/sitinc/journey-discovery-getting-started/blob/main/notes/) to explore usage.  


## Updates and Breaking Changes

This module is something I am putting together to allow everyone to have easy-to-use tools to analyze and understand their interactional data.  This project is not polished production code that has been battle hardened, but something I am building publicly as part of a blog series that will improve over time.  Sometimes, those changes are going to be breaking.  I'll aim for all the best practices around backwards compatibility once we hit v1.0.0.

Happy coding!  :)
