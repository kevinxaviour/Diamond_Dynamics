# Diamond Dynamics

### Project Takeaways:
- Data Cleaning & Preprocessing
- EDA & Data Visualization
- Feature Engineering
- Outlier & Skewness Handling
- Regression using ML algorithms, including ANN
- Dimensionality Reduction with PCA
- Clustering & Cluster Labeling
- Streamlit UI Design

## Work Flow
- Data Insertion and Preparation  [data_exploration.ipynb](https://github.com/kevinxaviour/Diamond_Dynamics/blob/5d032700a39103fbf55febeefc75ac3a5b00af42/data_exploration.ipynb)
  - Data Preprocessing
  - Feature Engineering
    - Handle x,y,z values which were 0
    - Derived new columns such as Volume,Dimension Ratio.etc
  - Encoding
    - Created Ordinal Encoding for columns Cut,Clarity and Color
    - Created Pickle file to dump the encoders
- Exploratory Data Analysis [EDA.ipynb](https://github.com/kevinxaviour/Diamond_Dynamics/blob/5d032700a39103fbf55febeefc75ac3a5b00af42/EDA.ipynb)
- Regression Model [Reg_model_training.ipynb](https://github.com/kevinxaviour/Diamond_Dynamics/blob/5d032700a39103fbf55febeefc75ac3a5b00af42/Reg_model_training.ipynb)
  - Created 9 different Regression Algorithms and Chose the best Model for Deployment to predict Diamond Prices
    - Linear Regression
    - KNN Regressor
    - Support Vector Regressor
    - Decision Tree Regressor
    - Random Forest Regressor
    - Ada Boost Regressor
    - Gradient Boost Regressor
    - Xg Boost Regressor
    - ANN
<img width="761" height="339" alt="image" src="https://github.com/user-attachments/assets/fda717ff-10ce-448c-b4c4-ff1e040e5b37" />


  - Saved the Best Performing Model in a Pickle File
- Clustering Model [cluster_model.ipynb](https://github.com/kevinxaviour/Diamond_Dynamics/blob/5d032700a39103fbf55febeefc75ac3a5b00af42/cluster_model.ipynb)
  - Used PCA for dimensionality reduction to 2 components
  - Created 5 different Clustering Model and chose the best model for deployment to cluster Diamond, then analyze average price, carat, and cut distribution per        cluster
    - K-Means Clustering
      - Performed Elbow Method to get the K-value 
    - Mini Batch K means Clustering
    - Hierarchical Clustering
    - Density Based Clustering
    - Gaussian Mixture
  - Saved The model with best Silhoutte Score in a Pickle File

<img width="401" height="91" alt="image" src="https://github.com/user-attachments/assets/6d27505d-d0d7-4aa4-9134-f65e739d84e8" />

- S3 Bucket 
  - Saved all the encoders and models in the S3 Bucket.
  <img width="1119" height="395" alt="image" src="https://github.com/user-attachments/assets/6d5e1923-56aa-429e-b141-88e62409271e" />


- Streamlit UI (Application)[https://diamonddynamics.streamlit.app/]
  - Retrieved all the files from S3 bucket for final predictions.
  - Created Input Features with sliders and number input boxes
  - Predicted the input values
  <img width="963" height="226" alt="image" src="https://github.com/user-attachments/assets/07f8e2de-4a12-49e3-86ed-c4c3f8766ff0" />
