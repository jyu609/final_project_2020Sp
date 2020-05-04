import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def read_covid_data() -> pd.DataFrame:
    """
    read the COVID-19 confirmed and death dataset, and return the combined dataset
    :return DataFrame of COVID-19 confirmed and death data

    """

    confirmed_dict = read_confirmed()
    death_dict = read_death()

    confirmed = pd.Series(confirmed_dict)
    death = pd.Series(death_dict)
    covid_data = pd.DataFrame(list(zip(confirmed, death)), columns=["confirmed", "death"])
    covid_data.insert(0, "Country", confirmed_dict.keys())

    return covid_data


def read_life_expectancy() -> pd.DataFrame:
    """
    read the life expectancy at birth dataset and modify some country names to the same as COVID-19 data
    :return DataFrame of life expectancy at birth data

    """

    life_df = pd.read_csv("data/API_SP.DYN.LE00.IN_DS2_en_csv_v2_988752.csv",
                          header=2, usecols=[0,62], names=["Country", "Life expectancy"])

    index = life_df[life_df["Country"]=="Iran, Islamic Rep."].index.values[0]
    life_df.loc[index, "Country"] = "Iran"
    index = life_df[life_df["Country"] == "United States"].index.values[0]
    life_df.loc[index, "Country"] = "US"
    index = life_df[life_df["Country"] == "Russian Federation"].index.values[0]
    life_df.loc[index, "Country"] = "Russia"

    # life expectancy of Dominica is NaN.
    # Since Dominica has same confirmed and death with Namibia, we assume they also have the same life expectancy
    domi_index = life_df[life_df["Country"] == "Dominica"].index.values[0]
    nami_index = life_df[life_df["Country"] == "Namibia"].index.values[0]
    life_df.loc[domi_index, "Life expectancy"] = life_df.loc[nami_index, "Life expectancy"]

    return life_df


def correlation_analysis():
    """
    main function of the correlation analysis

    """

    covid_data = read_covid_data()
    # print(covid_data)

    life_expectancy_data = read_life_expectancy()
    # print(life_expectancy_data)

    covid_life_joined = pd.merge(covid_data, life_expectancy_data, on="Country")
    # print(covid_life_joined)

    print("Correlation between 'confirmed' and 'life expectancy at birth':")
    print("Covariance: " + str(calculate_covariance(covid_life_joined["confirmed"], covid_life_joined["Life expectancy"])))
    print("Correlation coefficient: " + str(calculate_correlation_coefficient(covid_life_joined["confirmed"],
                                                                            covid_life_joined["Life expectancy"])))
    print()
    print("Correlation between 'death' and 'life expectancy at birth':")
    print("Covariance: " + str(calculate_covariance(covid_life_joined["death"], covid_life_joined["Life expectancy"])))
    print("Correlation coefficient: " + str(calculate_correlation_coefficient(covid_life_joined["death"],
                                                                              covid_life_joined["Life expectancy"])))

    draw_scatter_plot(covid_life_joined["confirmed"].head(12), covid_life_joined["Life expectancy"].head(12), "confirmed", "Life expectancy")
    draw_scatter_plot(covid_life_joined["death"].head(12), covid_life_joined["Life expectancy"].head(12), "death", "Life expectancy")


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

def draw_scatter_plot(x: pd.Series, y: pd.Series, x_label: str, y_label: str):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y)
    # plt.legend()
    plt.show()

if __name__ == '__main__':
    correlation_analysis()