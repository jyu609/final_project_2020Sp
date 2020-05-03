import pandas as pd
import numpy as np
import matplotlib as plt


def read_confirmed() -> dict:
    """
    Read the 'data/time_series_covid19_confirmed_global.csv' and extract data
    :return: the number of confirmed cases of the seleted countries
    """

    confirmed_dict = {'US': 0, 'Spain': 0, 'Italy': 0, 'United Kingdom': 0, \
                      'France': 0, 'Germany': 0, 'Turkey': 0, 'Russia': 0, 'Iran': 0, \
                      'Brazil': 0, 'China': 0, 'Canada': 0, 'Fiji': 0, 'Belize': 0, \
                      'Namibia': 0, 'Dominica': 0, 'Tajikistan': 0, 'Nicaragua': 0, \
                      'Seychelles': 0, 'Burundi': 0, 'Suriname': 0, 'Mauritania': 0, \
                      'Bhutan': 0, 'Comoros': 0}
    with open("data/time_series_covid19_confirmed_global.csv") as f:
        title = f.readline()
        line = f.readline()
        while (line):
            entities = line.split(',')
            country = entities[1]
            cur_num = int(entities[len(entities) - 1])
            if country in confirmed_dict.keys():
                confirmed_dict[country] += cur_num
            line = f.readline()
    return confirmed_dict


def read_death() -> dict:
    """
    Read the 'data/time_series_covid19_deaths_global.csv' and extract data
    :return: the number of death cases of the seleted countries
    """

    death_dict = {'US': 0, 'Spain': 0, 'Italy': 0, 'United Kingdom': 0, \
                  'France': 0, 'Germany': 0, 'Turkey': 0, 'Russia': 0, 'Iran': 0, \
                  'Brazil': 0, 'China': 0, 'Canada': 0, 'Fiji': 0, 'Belize': 0, \
                  'Namibia': 0, 'Dominica': 0, 'Tajikistan': 0, 'Nicaragua': 0, \
                  'Seychelles': 0, 'Burundi': 0, 'Suriname': 0, 'Mauritania': 0, \
                  'Bhutan': 0, 'Comoros': 0}
    with open("data/time_series_covid19_deaths_global.csv") as f:
        title = f.readline()
        line = f.readline()
        while (line):
            entities = line.split(',')
            country = entities[1]
            cur_num = int(entities[len(entities) - 1])
            if country in death_dict.keys():
                death_dict[country] += cur_num
            line = f.readline()
    return death_dict


def read_data() -> pd.DataFrame:
    """
    read the COVID-19 confirmed and death dataset, and return the combined dataset
    :return DataFrame of COVID-19 confirmed and death data

    """

    confirmed_dict = read_confirmed()
    death_dict = read_death()

    confirmed = pd.Series(confirmed_dict)
    death = pd.Series(death_dict)
    covid_data = pd.DataFrame(list(zip(confirmed, death)), columns=["confirmed", "death"], index=confirmed_dict.keys())

    return covid_data


def correlation_analysis():
    """
    main function of the correlation analysis

    """

    covid_data = read_data()

    print(covid_data)


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