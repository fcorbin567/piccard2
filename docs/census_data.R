# install.packages(tidyverse)
# install.packages('sf')
# install.packages("cancensus")
# install.packages("geojsonsf")
set_cancensus_api_key('CensusMapper_526e5fc2b21b8413cc4c0d5dd3aafea6', install = TRUE)

library(tidyverse)
library(sf)
library(cancensus)
library(geojsonsf)
options(cancensus.api_key = "CensusMapper_526e5fc2b21b8413cc4c0d5dd3aafea6")

all_vectors_06 = find_census_vectors('shelter',
                                dataset = 'CA06',
                                type = 'total',
                                query_type = 'semantic')
all_vectors_11 = find_census_vectors('housing',
                                 dataset = 'CA11',
                                 type = 'total',
                                 query_type = 'semantic')
all_vectors_16 = find_census_vectors('housing',
                                 dataset = 'CA16',
                                 type = 'total',
                                 query_type = 'semantic')
all_vectors_21 = find_census_vectors('housing',
                                 dataset = 'CA21',
                                 type = 'total',
                                 query_type = 'semantic')

selected_vectors_06 = c('v_CA06_102', 'v_CA06_103', 'v_CA06_2054', 
                        'v_CA06_2050', 'v_CA06_2057', 'v_CA06_2053', 
                        'v_CA06_2052', 'v_CA06_2049')
selected_vectors_11 = c('v_CA11N_2253', 'v_CA11N_2254', 'v_CA11N_2287',
                        'v_CA11N_2292', 'v_CA11N_2283', 'v_CA11N_2290')
selected_vectors_16 = c('v_CA16_4837', 'v_CA16_4838', 'v_CA16_4896',
                        'v_CA16_4901', 'v_CA16_4892', 'v_CA16_4899')
selected_vectors_21 = c('v_CA21_4238', 'v_CA21_4239', 'v_CA21_4312',
                        'v_CA21_4318', 'v_CA21_4307', 'v_CA21_4315')

to_census_06 = get_census(dataset = 'CA06',
                           regions = list(CMA='35535'),
                           vectors = selected_vectors_06,
                           level = 'CT',
                           use_cache = FALSE,
                           quiet = TRUE,
                           geo_format = 'sf')
to_census_11 = get_census(dataset = 'CA11',
                          regions = list(CMA='35535'),
                          vectors = selected_vectors_11,
                          level = 'CT',
                          use_cache = FALSE,
                          quiet = TRUE,
                          geo_format = 'sf')
to_census_16 = get_census(dataset = 'CA16',
                          regions = list(CMA='35535'),
                          vectors = selected_vectors_16,
                          level = 'CT',
                          use_cache = FALSE,
                          quiet = TRUE,
                          geo_format = 'sf')
to_census_21 = get_census(dataset = 'CA21',
                          regions = list(CMA='35535'),
                          vectors = selected_vectors_21,
                          level = 'CT',
                          use_cache = FALSE,
                          quiet = TRUE,
                          geo_format = 'sf')

st_write(to_census_06, "C:/Users/ecorb/OneDrive/Documents/school/Summer 2025/SUDS/piccard2/docs/piccard2_testing_data/housing_data_06.geojson", driver = "GeoJSON")
st_write(to_census_11, "C:/Users/ecorb/OneDrive/Documents/school/Summer 2025/SUDS/piccard2/docs/piccard2_testing_data/housing_data_11.geojson", driver = "GeoJSON")
st_write(to_census_16, "C:/Users/ecorb/OneDrive/Documents/school/Summer 2025/SUDS/piccard2/docs/piccard2_testing_data/housing_data_16.geojson", driver = "GeoJSON")
st_write(to_census_21, "C:/Users/ecorb/OneDrive/Documents/school/Summer 2025/SUDS/piccard2/docs/piccard2_testing_data/housing_data_21.geojson", driver = "GeoJSON")
