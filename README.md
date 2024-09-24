# Demo App for Fundamental Operations in Data Science

App deployed on Streamlit Cloud: https://dsg-app.streamlit.app

## Python Environment Setup and Management
**Install** conda environment:
```sh
$ conda env create -f conda.yml
```

Creates conda environment "dsgapp".

**Update** the environment with new packages/versions:
1. modify conda.yml
2. run `conda env update`:
```sh
$ conda env update --name sample --file conda.yml --prune
```
`prune` uninstalls dependencies which were removed from conda.yml

**Use** environment:
before working on the project always make sure you have the environment activated:
```sh
$ conda activate dsgapp
```

**Check the version** of a specific package (e.g. `html5lib`) in the environment:
```sh
$ conda list html5lib
```

**Export** an environment file across platforms:
Include only the packages that were specifically installed. Dependencies will be resolved upon installation
```sh
$ conda env export --from-history > conda.yml
```

**List** all installed environments:
From the base environment run
```sh
$ conda info --envs
```

**Remove** environment:
```sh
$ conda env remove -n dsgapp
```

See the complete documentation on [managing conda-environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## Runtime Configuration with Environment Variables
The environment variables are specified in a .env-file, which is never commited into version control, as it may contain secrets. The repo just contains the file `.env.template` to demonstrate how environment variables are specified.

You have to create a local copy of `.env.template` in the project root folder and the easiest is to just name it `.env`.

The content of the .env-file is then read by the pypi-dependency: `python-dotenv`. Usage:
```python
import os
from dotenv import load_dotenv
```

`load_dotenv` reads the .env-file and sets the environment variables:

```python
load_dotenv()
```
which can then be accessed:

```python
os.environ['SAMPLE_VAR']
```

## Tesseract for Text Extraction
On Mac:
```sh
brew install tesseract-lang
```
Installs all available languages


## Specifying dependencies for Deployment on Streamlit Cloud
Conda is still not very well supported on streamlit cloud. Therefore all dependencies for deployment are specified for `pipenv` in the `requirements.txt` file. For now I just found it the easiest for me to continue working with conda locally as the environment managager but rely on the pip-package manager. Before deployment I manually update `requirements.txt`

Upon deployment on Streamlit Cloud the Python version consistent with the local conda environment is specified manually (modal -> Advanced Settings): Python v. 3.10

tesseract in file `packages.txt` for ubuntu packages