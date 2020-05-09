import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import seaborn as sns


def read_covid_csv(path) -> dict:
    """
    Read the 'data/time_series_covid19_confirmed_global.csv' and extract data
    :param path: the path of the covid19 csv file
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
    covid_data = pd.DataFrame(list(zip(confirmed, death)), columns=["Confirmed", "Death"])
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

    life_df = life_df.dropna()

    return life_df


def read_population() -> pd.DataFrame:
    """
    read the population and modify some country names to the same as COVID-19 data
    :return DataFrame of population data

    """

    pop_df = pd.read_csv("data/API_SP.POP.TOTL_DS2_en_csv_v2_988606.csv",
                          header=2, usecols=[0,62], names=["Country", "Population"])

    index = pop_df[pop_df["Country"]=="Iran, Islamic Rep."].index.values[0]
    pop_df.loc[index, "Country"] = "Iran"
    index = pop_df[pop_df["Country"] == "United States"].index.values[0]
    pop_df.loc[index, "Country"] = "US"
    index = pop_df[pop_df["Country"] == "Russian Federation"].index.values[0]
    pop_df.loc[index, "Country"] = "Russia"

    pop_df = pop_df.dropna()

    return pop_df


def read_gdp() -> pd.DataFrame:
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


def read_education() -> pd.DataFrame:
    """
    read the Expected years of schooling (years).csv dataset and modify some country names to the same as COVID-19 data
    :return DataFrame of expected years of schooling

    """

    school_df = pd.read_csv("data/Expected years of schooling (years).csv", header=2, usecols=[1, 32], names=["Country", "Education"])

    index = school_df[school_df["Country"]=="Iran (Islamic Republic of)"].index.values[0]
    school_df.loc[index, "Country"] = "Iran"
    index = school_df[school_df["Country"] == "United States"].index.values[0]
    school_df.loc[index, "Country"] = "US"
    index = school_df[school_df["Country"] == "Russian Federation"].index.values[0]
    school_df.loc[index, "Country"] = "Russia"

    school_df = school_df.dropna()

    return school_df


def read_internet() -> pd.DataFrame:
    """
    read the Internet users, total (% of population) dataset and modify some country names to the same as COVID-19 data
    :return DataFrame of the internet users percentage

    """

    internet_df = pd.read_csv("data/Internet users, total (% of population).csv", header=2, usecols=[1, 20], names=["Country", "Internet"])

    index = internet_df[internet_df["Country"]=="Iran (Islamic Republic of)"].index.values[0]
    internet_df.loc[index, "Country"] = "Iran"
    index = internet_df[internet_df["Country"] == "United States"].index.values[0]
    internet_df.loc[index, "Country"] = "US"
    index = internet_df[internet_df["Country"] == "Russian Federation"].index.values[0]
    internet_df.loc[index, "Country"] = "Russia"

    internet_df = internet_df.dropna()

    return internet_df

def correlation_analysis():
    """
    main function of the correlation analysis

    """

    raw_covid_data = read_covid_data()

    pop_data = read_population()

    life_expectancy_data = read_life_expectancy()

    gdp_data = read_gdp()

    edu_data = read_education()

    int_data = read_internet()

    covid_joined = pd.merge(raw_covid_data, pop_data, on="Country")

    covid_joined.insert(4, "Confirmed rate", covid_joined["Confirmed"] / covid_joined["Population"])
    covid_joined.insert(5, "Death rate", covid_joined["Death"] / covid_joined["Population"])

    covid_life_joined = pd.merge(covid_joined, life_expectancy_data, on="Country")
    covid_life_gdp_joined = pd.merge(covid_life_joined, gdp_data, on="Country")
    covid_life_gdp_edu_joined = pd.merge(covid_life_gdp_joined, edu_data, on="Country")
    covid_life_gdp_edu_int_joined = pd.merge(covid_life_gdp_edu_joined, int_data, on="Country")
    covid_life_gdp_edu_int_joined = covid_life_gdp_edu_int_joined[covid_life_gdp_edu_int_joined.Education != '..']
    covid_life_gdp_edu_int_joined = covid_life_gdp_edu_int_joined[covid_life_gdp_edu_int_joined.Internet != '..']
    covid_life_gdp_edu_int_joined['Education'] = covid_life_gdp_edu_int_joined['Education'].astype(float)
    covid_life_gdp_edu_int_joined['Internet'] = covid_life_gdp_edu_int_joined['Internet'].astype(float)

    sns.set()

    draw_histogram(covid_life_gdp_edu_int_joined["Confirmed rate"], "COVID-19 Confirmed rate")
    draw_histogram(covid_life_gdp_edu_int_joined["Death rate"], "COVID-19 Death rate")

    display_analysis_result(covid_life_gdp_edu_int_joined["Life expectancy"], covid_life_gdp_edu_int_joined["Confirmed rate"], "Life expectancy", "Confirmed rate")
    display_analysis_result(covid_life_gdp_edu_int_joined["Life expectancy"], covid_life_gdp_edu_int_joined["Death rate"], "Life expectancy", "Death rate")

    display_analysis_result(covid_life_gdp_edu_int_joined["GDP"], covid_life_gdp_edu_int_joined["Confirmed rate"], "GDP", "Confirmed rate")
    display_analysis_result(covid_life_gdp_edu_int_joined["GDP"], covid_life_gdp_edu_int_joined["Death rate"], "GDP", "Death rate")

    display_analysis_result(covid_life_gdp_edu_int_joined["Education"], covid_life_gdp_edu_int_joined["Confirmed rate"], "Education", "Confirmed rate")
    display_analysis_result(covid_life_gdp_edu_int_joined["Education"], covid_life_gdp_edu_int_joined["Death rate"], "Education",  "Death rate")

    display_analysis_result(covid_life_gdp_edu_int_joined["Internet"], covid_life_gdp_edu_int_joined["Confirmed rate"], "Internet", "Confirmed rate")
    display_analysis_result(covid_life_gdp_edu_int_joined["Internet"], covid_life_gdp_edu_int_joined["Death rate"], "Internet", "Death rate")


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
    """
    calcualte the p-value of the significance test of the two dataframe columns
    :param column1: one column of a dataframe
    :param column2: one column of a dataframe
    :return p-value of the significance test of the two input columns

    >>> test_df = pd.DataFrame({"A":[1,2,3,4,10,34], "B": [5,6,7,9,38,78]})
    >>> calculate_significance_of_coefficient(test_df.A, test_df.B)
    0.0006257038151347064

    """

    p_value = stats.pearsonr(column1,column2)[1]
    return p_value


def draw_histogram(x: pd.Series, x_label: str):
    """
    Generate a histogram for the input variable
    :param x: one column of a dataframe
    :param x_label: name of x

    """

    x.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
    plt.title('Histogram for %s' % x_label)
    plt.show()


def draw_scatter_plot(x: pd.Series, y: pd.Series, x_label: str, y_label: str):
    """
    Generate a scatter plot based on two given dataframe columns, and draw its linear regression line
    :param x: one column of a dataframe
    :param y: one column of a dataframe
    :param x_label: name of x
    :param y_label: name of y

    """

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y)
    plt.title("Scatter plot of '%s' and '%s'" % (x_label, y_label))

    lr_model = linear_regression(x, y)
    plt.plot(x, lr_model.predict(np.array(x).reshape(-1, 1)), color='red')

    plt.show()


def draw_residual_plot(x: pd.Series, y: pd.Series, x_label: str, y_label: str):
    """
    Generate a residual plot based on two given dataframe columns
    :param x: one column of a dataframe
    :param y: one column of a dataframe
    :param x_label: name of x
    :param y_label: name of y

    """

    plt.title("Residual plot of '%s' and '%s'" % (x_label, y_label))
    sns.residplot(x=x, y=y, scatter_kws={"s": 80})
    plt.show()


def display_analysis_result(column1: pd.Series, column2: pd.Series, name1: str, name2: str):
    """
    Display the analysis result based on two given dataframe columns
    :param column1: one column of a dataframe
    :param column2: one column of a dataframe
    :param name1: name of x
    :param name2: name of y

    """

    print("Correlation between '%s' and '%s':" % (name1, name2))
    print("Covariance: " + str(calculate_covariance(column1, column2)))
    print("Correlation coefficient: " + str(calculate_correlation_coefficient(column1, column2)))
    print("Significance of coefficient: " + str(calculate_significance_of_coefficient(column1, column2)))
    print()

    draw_scatter_plot(column1, column2, name1, name2)
    draw_residual_plot(column1, column2, name1, name2)


def linear_regression(x: pd.Series, y: pd.Series) -> LinearRegression:
    """
    Create the linear regression model using the two input variables
    :param column1: one column of a dataframe
    :param column2: one column of a dataframe
    :return linear regression model

    """

    lr_model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    lr_model.fit(np.array(x).reshape(-1,1), y)

    return lr_model


if __name__ == '__main__':

    correlation_analysis()
