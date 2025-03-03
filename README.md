# Bank Customer Churn Prediction

## Project Overview
This project analyzes customer data from a bank to predict which customers are likely to leave (churn). Customer churn is a critical business metric, as retaining existing customers is typically more cost-effective than acquiring new ones. By identifying customers at risk of leaving, banks can take proactive steps to improve retention.

## Dataset
The analysis uses the "Churn_Modelling.csv" dataset, obtained from open source Kaggle, which contains information about 10,000 bank customers with the following features:
- Basic customer information (Customer ID, Surname)
- Demographics (Credit Score, Geography, Gender, Age)
- Banking relationship (Tenure, Balance, Number of Products, Credit Card ownership, Active membership status)
- Financial data (Estimated Salary)
- Target variable: "Exited" (1 = customer left the bank, 0 = customer stayed)

## Tools Used
- **Python**: Primary programming language for analysis
- **Jupyter Notebook**: Interactive development environment for code execution and documentation
- **Pandas**: Data manipulation and analysis library
- **NumPy**: Numerical computing library for mathematical operations
- **Matplotlib & Seaborn**: Data visualization libraries for creating plots and charts
- **Scikit-learn**: Machine learning library providing tools for predictive modeling
  - RandomForestClassifier
  - LogisticRegression
  - SVC (Support Vector Classification)
  - KNeighborsClassifier
  - GradientBoostingClassifier
  - StandardScaler for feature normalization
  - Classification metrics (confusion_matrix, classification_report, accuracy_score)
- **LabelEncoder**: Tool for converting categorical data to numerical format

## Project Steps

### 1. Data Exploration and Cleaning
- Checked for missing values (none found)
- Verified no duplicate records exist
- Examined data types and structure

### 2. Exploratory Data Analysis (EDA)
The analysis included several visualizations to understand patterns in customer churn:

- **Overall Churn Distribution**: Visual representation of how many customers stayed vs. left
- **Churn by Geography**: Comparison of churn rates across different countries
- **Churn by Gender**: Analysis of whether gender impacts likelihood to leave
- **Age Distribution by Churn**: Visualization showing how age relates to churn behavior
- **Correlation Heatmap**: Identification of relationships between numeric variables
- **Financial Analysis**: Examination of how balance and salary relate to churn
- **Credit Score Analysis**: Comparison of credit scores between churned and retained customers

Key insights from these visualizations help identify patterns that might predict customer churn.

### 3. Data Preprocessing
Before building predictive models, the data was prepared through:
- Converting categorical variables (Gender, Geography) to numerical format
- Feature selection to choose relevant predictors
- Splitting data into training (80%) and testing (20%) sets
- Standardizing features to ensure all variables are on a similar scale

### 4. Model Building and Evaluation
Several machine learning models were trained and compared:

1. **Random Forest Classifier** (87% accuracy)
   - Strong overall performer
   - Good balance of precision and recall

2. **Logistic Regression** (81% accuracy)
   - Simple interpretable model
   - Performed reasonably well but struggled with identifying churned customers

3. **Support Vector Machine (SVM)** (80% accuracy)
   - Failed to identify any churned customers
   - Good for identifying non-churned customers but not useful as a complete solution

4. **K-Nearest Neighbors (KNN)** (82% accuracy)
   - Moderate performance
   - Better than logistic regression at identifying churned customers

5. **Gradient Boosting** (87% accuracy)
   - On par with Random Forest
   - Strong overall performance

Each model was evaluated using:
- Confusion Matrix: Shows true positives, false positives, true negatives, and false negatives
- Classification Report: Details precision, recall, and F1-score for each class
- Accuracy Score: Overall percentage of correct predictions

### 5. Feature Engineering
Additional features were created to potentially improve model performance:
- **BalanceZero**: Flag for customers with zero balance
- **AgeGroup**: Categorized age into meaningful groups
- **BalanceToSalaryRatio**: Ratio of balance to salary
- **ProductUsage**: Interaction between number of products and active membership
- **TenureGroup**: Categorized tenure into groups
- **Gender-Geography Interactions**: Combined gender with country information

Despite these additional features, model accuracy remained similar at 87%, indicating that the original features already captured most of the predictive information.

### 6. Feature Importance Analysis
The Random Forest model identified the most important features for predicting churn:
- Age
- Balance
- Estimated Salary
- Geography
- Active membership status

## Conclusions
- The Random Forest and Gradient Boosting models performed best with 87% accuracy
- Age appears to be a significant factor in customer churn
- The models could identify customers who stay with high accuracy (96% recall) but were less effective at identifying customers who leave (46-49% recall)
- Feature engineering did not significantly improve model performance

## Business Recommendations
1. Develop retention strategies focused on older customers
2. Pay special attention to customers with certain balance profiles
3. Consider geography-specific retention programs
4. Implement programs to increase active membership status
5. Focus retention efforts on the specific customer segments identified by the model

## Technical Requirements
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

## Running the Analysis
1. Ensure all required libraries are installed
2. Place the "Churn_Modelling.csv" file in the same directory as the notebook
3. Run the Jupyter notebook cells in sequence
