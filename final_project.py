import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def read_covid_csv(path) -> dict:
    """
    Read the 'data/time_series_covid19_confirmed_global.csv' and extract data
    :return: the number of confirmed cases of the seleted countries
    """

    dict = {}
    with open(path) as f:
        title = f.readline()
        line = f.readline()
        while (line):
            entities = line.split(',')
            country = entities[1]
            cur_num = int(entities[len(entities) - 1])
            if country in dict.keys():
                dict[country] += cur_num
            else:
                dict[country] = cur_num
            line = f.readline()
    return dict


def read_covid_data() -> pd.DataFrame:
    """
    read the COVID-19 confirmed and death dataset, and return the combined dataset
    :return DataFrame of COVID-19 confirmed and death data

    """

    confirmed_dict = read_covid_csv("data/time_series_covid19_confirmed_global.csv")
    death_dict = read_covid_csv("data/time_series_covid19_deaths_global.csv")

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
    # domi_index = life_df[life_df["Country"] == "Dominica"].index.values[0]
    # nami_index = life_df[life_df["Country"] == "Namibia"].index.values[0]
    # life_df.loc[domi_index, "Life expectancy"] = life_df.loc[nami_index, "Life expectancy"]

    life_df = life_df.dropna()

    return life_df


def read_GDP() -> pd.DataFrame:
    """
    read the GDP at birth dataset and modify some country names to the same as COVID-19 data
    :return DataFrame of GDP at birth data

    """

    gdp_df = pd.read_csv("data/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_988471.csv",
                          header=4, usecols=[0,62], names=["Country", "GDP"])

    index = gdp_df[gdp_df["Country"]=="Iran, Islamic Rep."].index.values[0]
    gdp_df.loc[index, "Country"] = "Iran"
    index = gdp_df[gdp_df["Country"] == "United States"].index.values[0]
    gdp_df.loc[index, "Country"] = "US"
    index = gdp_df[gdp_df["Country"] == "Russian Federation"].index.values[0]
    gdp_df.loc[index, "Country"] = "Russia"

    gdp_df = gdp_df.dropna()

    return gdp_df


def correlation_analysis():
    """
    main function of the correlation analysis

    """

    covid_data = read_covid_data()
    # print(covid_data)

    life_expectancy_data = read_life_expectancy()
    # print(life_expectancy_data)

    gdp_data = read_GDP()
    # print(gdp_data)

    covid_life_joined = pd.merge(covid_data, life_expectancy_data, on="Country")
    # print(covid_life_joined)

    covid_life_gdp_joined = pd.merge(covid_life_joined, gdp_data, on="Country")
    # print(covid_life_gdp_joined)

    display_analysis_result(covid_life_gdp_joined["confirmed"], covid_life_gdp_joined["Life expectancy"], "confirmed", "life expectancy")
    display_analysis_result(covid_life_gdp_joined["death"], covid_life_gdp_joined["Life expectancy"], "death", "life expectancy")

    display_analysis_result(covid_life_gdp_joined["confirmed"], covid_life_gdp_joined["GDP"], "confirmed", "GDP")
    display_analysis_result(covid_life_gdp_joined["death"], covid_life_gdp_joined["GDP"], "death", "GDP")


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


def calculate_significance_of_coefficient(column1: pd.Series, column2: pd.Series) -> np.float64:
    p_value = stats.pearsonr(column1,column2)[1]
    return p_value


def draw_scatter_plot(x: pd.Series, y: pd.Series, x_label: str, y_label: str):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y)
    # plt.legend()
    plt.show()


def display_analysis_result(column1: pd.Series, column2: pd.Series, name1: str, name2: str):
    print("Correlation between '%s' and '%s':" % (name1, name2))
    # covariance is positive, so they have positive relation
    print("Covariance: " + str(calculate_covariance(column1, column2)))
    # correlation coefficient is between 0.4 and 0.8, they have moderately linear correlation
    print("Correlation coefficient: " + str(calculate_correlation_coefficient(column1, column2)))
    # p-value is less than 0.05, indicating a significant correlation
    print("Significance of coefficient: " + str(calculate_significance_of_coefficient(column1, column2)))
    print()

    draw_scatter_plot(column1, column2, name1, name2)


if __name__ == '__main__':
    correlation_analysis()