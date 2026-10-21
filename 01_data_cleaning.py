import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# importing datasets
edu_d = pd.read_csv("us_census_dataset/Educationv.csv")
fin_d = pd.read_csv("us_census_dataset/Finance.csv")
ind_d = pd.read_csv("us_census_dataset/Industry.csv")

# dictionaries for renaming, which we can use to form codebook later
edu_rename_dict = {
	"Year": "year",
	"Bachelors_degree_or_higher": "collegeGrad",
	"high_school_or_some_degree": "highGrad",
	"Less_than_high_school_graduate": "noGrad"
}

ind_rename_dict = {
	"Year": "year",
	"Total_Agriculture_forestry_fishing_hunting_mining": "totalNaturalRes",
	"Total_Construction": "totalConstruction",
	"Total_Manufacturing": "totalManufacturing",
	"Total_Wholesale_trade": "totalWholesaleTrade",
	"Total_Retail_trade": "totalRetailTrade",
	"Total_Transportation_warehousing_utilities": "totalTranspUtil",
	"Total_Information": "totalInfo",
	"Total_Finance_insurance_realestate_rental_leasing": "totalFinancial",
	"Total_Professional_scientific_management_administrative_waste_management_services": "totalProfessional",
	"Total_Educationalservices_healthcare_socialassistance": "totalSocial",
	"Total_Arts_entertainment_recreation_accommodation_foodservices": "totalArts",
	"Total_Otherservices_except_Public_administration": "totalOther",
	"Total_Public_administration": "totalPublicAdmin",
	# Are we using gender values? I believe we said NO in our paper
	"Male": "male",
	"Male_Agriculture_forestry_fishing_hunting_mining": "maleNaturalRes",
	"Male_Construction": "maleConstruction",
	"Male_Manufacturing": "maleManufacturing",
	"Male_Wholesale_trade": "maleWholesaleTrade",
	"Male_Retail_trade": "maleRetailTrade",
	"Male_Transportation_warehousing_utilities": "maleTranspUtil",
	"Male_Information": "maleInfo",
	"Male_Finance_insurance_realestate_rental_leasing": "maleFinancial",
	"Male_Professional_scientific_management_administrative_waste_management_services": "maleProfessional",
	"Male_Educationalservices_healthcare_socialassistance": "maleSocial",
	"Male_Arts_entertainment_recreation_accommodation_foodservices": "maleArts",
	"Male_Otherservices_except_Public_administration": "maleOther",
	"Male_Public_administration": "malePublicAdmin",
	"Female": "female",
	"Female_Agriculture_forestry_fishing_hunting_mining": "femaleNaturalRes",
	"Female_Construction": "femaleConstruction",
	"Female_Manufacturing": "femaleManufacturing",
	"Female_Wholesale_trade": "femaleWholesaleTrade",
	"Female_Retail_trade": "femaleRetailTrade",
	"Female_Transportation_warehousing_utilities": "femaleTranspUtil",
	"Female_Information": "femaleInfo",
	"Female_Finance_insurance_realestate_rental_leasing": "femaleFinancial",
	"Female_Professional_scientific_management_administrative_waste_management_services": "femaleProfessional",
	"Female_Educationalservices_healthcare_socialassistance": "femaleSocial",
	"Female_Arts_entertainment_recreation_accommodation_foodservices": "femaleArts",
	"Female_Otherservices_except_Public_administration": "femaleOther",
	"Female_Public_administration": "femalePublicAdmin"
}

fin_rename_dict = {
	"Year": "year"
}

# data cleaning
edu_c = edu_d.rename(columns = edu_rename_dict)
ind_c = ind_d.rename(columns = ind_rename_dict)
fin_c = fin_d.rename(columns = fin_rename_dict) 
# cleaning fin data...
# What was decided as the cutoff for "is poor" in a district?
# Ideally, here that would be computed
# I find it reasonable to combine all of the variables below it into one, and all of the variables above it into one.
# Then a proportional value can be calculated?

# for removing gendered variables
ind_c = ind_c.loc[:, ~ind_c.columns.str.startswith(('male', 'female'))]			

# ---------------------
# Calculating poverty proportional data using a cutoff of





# ---------------------
# Joining cleaned tables
# We join the tables on year as the 2019 year is not reflected in the education data available. Then, we need to trim this year from finance and industry through this join.

cens_c = edu_c.merge(ind_c, on=["year", "cd"])
cens_c = cens_c.merge(fin_c, on=["year", "cd"])

print(cens_c.columns)

# --------------------- 
# Here we split the data into training and validation sets using a standard 80:20 propotion

# fcol = [
	
	
