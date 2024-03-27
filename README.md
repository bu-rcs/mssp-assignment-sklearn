# Assignment Scikit-Learn

This assignment is written in Python using sklearn. It is tested with Pytest.

## The assignment

For this assignment you will again work with with regular season NBA player
statistics. There are now 2 datasets. One from 2022-2023 regular season and one
for the current 2023-2024 season. The 2022-203 dataset has an additional column
indicating whether a player was nominated for the Allstar game, 1 for yes, and 0
for no. You will use these datasets to accomplish the following tasks.

1. Write a function that:
    - Imports the 2022-2023 and 2023-2024 regular season datasets.
    - Trains a random forest classifier to predict whether a player made the
    Allstar game. The feature columns should be: Points, Assists, Rebounds,
    Steals, Blocks, and Turnovers. The target column is whether the player is an
    Allstar.
    - Predicts whether a player from the 2023-2024 dataset made the Allstar game.

    This function should return a dataframe with the Player names, input
    features, and a column called Allstar Prediction with the predictions of the
    Random Forest Model.
2. In the dataset directory there is a csv file with the actual list of the
    players who made the 2023-2024 Allstar game. Parse this csv file and 
    evaluate the performance of your model. Your function should
    return a dictionary containing all the information in your classification
    report.

## Run command

On the SCC:

```bash
module load miniconda
module load academic-ml/spring-2024
conda activate spring-2024-pyt
cd /path/that/contains/this/repo
pytest
```
