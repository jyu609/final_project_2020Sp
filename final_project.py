import pandas as pd
import numpy as np

confirmed_dict = {}
death_dict = {}

def read_confirmed():
    with open("data/time_series_covid19_confirmed_global.csv") as f:
        title = f.readline()
        line = f.readline()
        while(line):
            entities = line.split(',')
            country = entities[1]
            cur_num = int(entities[len(entities) - 1])
            if country in confirmed_dict.keys():
                confirmed_dict[country] += cur_num
            else:
                confirmed_dict[country] = cur_num
            line = f.readline()
        print(confirmed_dict)

def read_death():
    with open("data/time_series_covid19_deaths_global.csv") as f:
        title = f.readline()
        line = f.readline()
        while(line):
            entities = line.split(',')
            country = entities[1]
            cur_num = int(entities[len(entities) - 1])
            if country in death_dict.keys():
                death_dict[country] += cur_num
            else:
                death_dict[country] = cur_num
            line = f.readline()
        print(death_dict)

def correlation_analysis():
    print("start analysis")


def calculate_covariance(column1: pd.Series, column2: pd.Series) -> np.float64:
    """
    calcualte the covariance of the two dataframe columns
    :param column1: one column of a dataframe
    :param column2: one column of a dataframe
    :return covariance of the two input columns

    >>> test_df = pd.DataFrame({"A":[1,2,3,4], "B": [5,6,7,8]})
    >>> calculate_covariance(test_df.A, test_df.B)
    1.6666666666666665

    """

    cov = column1.cov(column2)
    return cov


def calculate_correlation_coefficient(column1: pd.Series, column2: pd.Series) -> np.float64:
    """
    calcualte the correlation coefficient of the two dataframe columns
    :param column1: one column of a dataframe
    :param column2: one column of a dataframe
    :return correlation coefficient of the two input columns

    >>> test_df = pd.DataFrame({"A":[1,2,3,4], "B": [5,6,7,8]})
    >>> calculate_correlation_coefficient(test_df.A, test_df.B)
    1.0

    """

    corr = column1.corr(column2)
    return corr


if __name__ == '__main__':
    correlation_analysis()
    read_confirmed()
    read_death()
