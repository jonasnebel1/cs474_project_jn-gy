import numpy as np
import pandas as pd

# importing datasets
edu_data = pd.read_csv("us_census_dataset/Educationv.csv")
fin_data = pd.read_csv("us_census_dataset/Finance.csv")
ind_data = pd.read_csv("us_census_dataset/Industry.csv");

#TODO Trim off data from 2019 in EDU and FINANCIAL dataframes.

print("###### Educational data ######")
print(edu_data.columns)
print(edu_data.head())
print(len(edu_data))

print("###### Financial data ######")
print(fin_data.columns)
print(fin_data.head())
print(len(fin_data))

print("###### Industry data ######")
print(ind_data.columns)
print(ind_data.head())
print(len(ind_data))

print("###### Merged Data ######")

census_data = edu_data.merge(fin_data, on=["Year", "cd"])
census_data = census_data.merge(ind_data, on=["Year", "cd"])
print(census_data.head())
print(census_data.columns)
print(len(census_data))
print(census_data.isna().sum())


print ("###### Latex Code ######")

print(fin_data.head().to_latex(index=False, float_format="{:.1f}".format))
print(edu_data.head().to_latex(index=False, float_format="{:.1f}".format))
print(ind_data.head().to_latex(index=False, float_format="{:.1f}".format))

trunc_ind_data = ind_data.iloc[:5, :5].copy()
trunc_ind_data['...'] = '...'
print(trunc_ind_data.head().to_latex(index=False, float_format="{:.1f}".format))
