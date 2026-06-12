

**Data Exploration** 



After Exploring Datasets On Kaggle I have Found A Good Matching dataset With my project "House Price Prediction dataset.csv" Of 91kb. This Dataset have 2000 rows and 10 column.

Column names are Id, Area, Bathroom, Bedroom, Floors, YearBuilt,  Location, Condition, Garage, Price.





**Data Preprocessing** 

I have chosen Kaggle as a Platform to Work on model, for Preprocessing, training, testing etc. When i applied encoding and preprocessing the LR was not running good, the R2 score Was worst 

\--- NEW MODEL METRICS ---

Mean Absolute Error (MAE): $243,453.00

Root Mean Squared Error (RMSE): $280,057.77

R-squared Score (R²): -0.0081


I fixed data contamination, datatype mismatches, coloumn spelling inconsistency, data alignment, changes in scaling
so I after trying everything i have found out that Dataset is Not aunthentic, it has fake data so i have changed the dataset 



**Data Exploration**



After Exploring Datasets On Kaggle I have Found A Good Matching dataset With my project "house\_pricing\_dataset.csv" Of 10kb. This Dataset have 121 rows and 18 columns.

Column names are house\_id	city	neighborhood	property\_type	bedrooms	bathrooms	area\_sqft	lot\_sqft	year\_built	garage\_spaces	has\_garden	

has\_pool	floors	school\_rating	crime\_rate	distance\_to\_cbd\_km	renovation\_score	energy\_rating	sale\_price\_usd.



**Data Preprocessing**

now the Evaluation metrics are :



========================================

     FINAL MODEL PERFORMANCE METRICS    

=========================================

Mean Absolute Error (MAE):     $ 13,633.85

Root Mean Squared Error (RMSE): $ 17,292.32

R-squared Score (R²):           0.9927


i have dropped some unnecessary columns and trained the model once again, my purpose of dropping unnecassar columns was to make the model fast and light. the existing columns are 
city	neighborhood	property_type	bedrooms	bathrooms	area_sqft	lot_sqft	year_built	garage_spaces	has_garden	has_pool	floors	sale_price_usd

and current metrics are :
=========================================
      FINAL MODEL PERFORMANCE METRICS    
=========================================
Mean Absolute Error (MAE):     $ 25,723.87
Root Mean Squared Error (RMSE): $ 30,397.36
R-squared Score (R²):           0.9774
=========================================



