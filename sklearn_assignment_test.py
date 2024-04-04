import sklearn_assignment as ska
import pandas as pd
import pytest


def test_problem1():
    df = ska.problem1("dataset/2022-2023_NBA_Player_Stats_Regular_Allstar.csv", 
                      "dataset/2023-2024_NBA_Player_Stats_Regular.csv", 
                      42)
    
    assert(isinstance(df, pd.DataFrame))
    assert(len(df) == 522)
    assert(df["Player"].loc[1] == "Bam Adebayo")
    assert(df["PTS"].loc[1] == pytest.approx(22.1, abs=0.5))
    assert(df["Allstar Prediction"].loc[1] == 1)


def test_problem2():
    report = ska.problem2("dataset/2022-2023_NBA_Player_Stats_Regular_Allstar.csv", 
                          "dataset/2023-2024_NBA_Player_Stats_Regular.csv", 
                          "dataset/2023_2024_Allstars.csv",
                          42)
    
    assert(isinstance(report, dict))
    assert(report["Allstar"]["precision"] == pytest.approx(0.61, abs=0.05))
    assert(report["accuracy"] == pytest.approx(0.97, abs=0.03))
