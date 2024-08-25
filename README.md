# Movie Sentiment Analaysis Project
The purpose of this project is to familiarize myself with industry-standard machine learning tools. 
My goal is to analyze movie reviews and find out the sentiment about them (Positive, Negative, Neutral).


# Contributing

Minimum Requirements:
- Python3 (version 3.10^)

## Installation steps

Installing and setup the Poetry for dependency management in our Python project
```bash
pipx install poetry

poetry config virtualenvs.in-project true # for developing in VSCode ... https://stackoverflow.com/questions/59882884/vscode-doesnt-show-poetry-virtualenvs-in-select-interpreter-option

poetry install
```


Running the project for the first time:

```bash
poetry run python3 src/ml.py --pretrained=False # When running for the first time, train the model
```

The pre-trained flag will indicate to the program that the model needs to be (re)generated. Once completed,
you will be prompted for input for the model to analyze the sentiment of your text.

On subsequent runs, if you don't want to regenerate the model, you can simply run the python script as-is:

```bash
poetry run python3 src/ml.py # the --pretrained flag has a default value of True
```