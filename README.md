# 590PR Final Project

## Group Member (NetID) and responsible part: 
Yilin Tan (yilint2), responsible for data processing of Life expectancy, drawing plots and making analysis

Pengxin Bian (pengxin2), responsible for data processing of GDP and writing dostests

Jingtao Yu (jyu69), responsible for data processing of Education and Internet, implement parallel programming and writing README and presentation slides

## Type of project: 
(Type II Projects) Specifics for an Original Data Analysis [Non-simulation]

## Title: 
Analysis of the relationship between the extent of COVID-19 and the developed level of countries

## Description: 
Our goal is to find out the relationship between the extent of COVID-19 and the developed level of countries. We use two factors to indicate the extent of COVID-19, the confirmed rate, and death rate. And we use factors in four aspects to reflect the developed level, economy (GDP per capita), education (excepted year of schooling), lifespans (life expectancy at birth), mobility and communication (Internet user percentage of a country).

## Hypothesis:

1.The GDP per capita has positive linear relationship with COVID-19 confirmed rate and death rate

2.The expected year of schooling has positive linear relationship with COVID-19 confirmed rate and death rate

3.The life expectancy at birth has positive linear relationship with COVID-19 confirmed rate and death rate

4.The Internet user percentage of a country has positive linear relationship with COVID-19 confirmed rate and death rate
## Analysis:
![](corrlation.png)
![](hist.png)
![](residual.png)
![](scatter.png)
## Conclusion:
* All our four hypothesis are supported. 
* That means that countries  with higher developed level tend to  have more serious extend of COVID-19 
* We think this may because: 
  * Testing problems in less developed countries.  
  * Low connectivity in less developed countries


## Data Source:

COVID-19 confirmed case and death case: https://github.com/CSSEGISandData/COVID-19

GDP per capita: https://data.worldbank.org/indicator/NY.GDP.PCAP.CD

Expected year of schooling & Internet user percentage : http://hdr.undp.org/en/data#

Life expectancy at birth: https://data.worldbank.org/indicator/SP.DYN.LE00.IN






