# About this project

This is the code repository for a project that aims to test whether, how, and how well humans encode values in their memory. We test the normative theory for allocating limited memory resources in reinforcement learning, which was originally originally proposed in [Patel et al. 2020, NeurIPS](https://papers.nips.cc/paper/2020/hash/c4fac8fb3c9e17a2f4553a001f631975-Abstract.html).

> :warning: **The data is not hosted on github.** Hence, none of the files in src/analysis will work unless you have your own data. If none of the collaborators ([Luigi Acerbi](https://luigiacerbi.com/), [Alexandre](https://neurocenter-unige.ch/research-groups/alexandre-pouget/) [Pouget](https://neurocenter-unige.ch/research-groups/alexandre-pouget/), and [Antonio](https://neurocenter-unige.ch/research-groups/alexandre-pouget/) [Rangel](https://www.rnl.caltech.edu/)) have any objections, we will release the data when our work gets published.

# Installation

We use [Poetry](https://python-poetry.org/), a modern python packaging and dependency management software. If you are unfamiliar with it, [here's a quick tutorial](https://www.youtube.com/watch?v=0f3moPe_bhk). To get started, you can use the following commands:

```sh
# clone the git repository and jump into the project folder
gh repo clone nisheetpatel/human-memory
cd human-memory

# install poetry if you don't have it already
pip install poetry

# configure virtual env to be created within the project
poetry config virtualenvs.in-project true

# create the environment and install dependencies
poetry install

# activate the virtual environment
poetry shell
```
