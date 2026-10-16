import numpy as np
import pandas as pd

# importing datasets, again
edu = pd.read_csv("us_census_dataset/Educationv.csv")
fin = pd.read_csv("us_census_dataset/Finance.csv")
ind = pd.read_csv("us_census_dataset/Industry.csv")

# Joining tables
# We join the tables on year as the 2019 year is not reflected in the education data available. Then, we need to trim this year from finance and industry through this join.

cens_d = edu.merge(fin, on ["Year", "cd"])
cens_d = cens_d.merge(ind, on=["Year, "cd"])


