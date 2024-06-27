# CUSTID: Identification of Credit Cardholder (Categorical)
# BALANCE: Balance amount left in their account to make purchases
# BALANCEFREQUENCY: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
# PURCHASES: Amount of purchases made from the account
# ONEOFFPURCHASES: Maximum purchase amount did in one-go
# INSTALLMENTSPURCHASES: Amount of purchase done in installment
# CASH ADVANCE: Cash in advance given by the user
# PURCHASESFREQUENCY: How frequently the Purchases are being made score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
# ONEOFFPURCHASESFREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
# PURCHASESINSTALLMENTSFREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
# CASHADVANCEFREQUENCY: How frequently the cash in advance being paid
# CASHADVANCETRX: Number of Transactions made with “Cash in Advanced”
# PURCHASESTRX: Number of purchase transactions made
# CREDIT LIMIT: Limit of Credit Card for user
# PAYMENTS: Amount of Payment done by the user
# MINIMUM_PAYMENTS: Minimum amount of payments made by the user
# PRCFULLPAYMENT: Percent of full payment paid by the user
# TENURE: Tenure of credit card service for user

#!/usr/bin/env python3

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# print(sys.executable)  # This prints the path of the Python interpreter being used.
# print(sys.path)        # This prints the Python path to check where Python is looking for packages.


# Load data from the downloaded CSV file
df = pd.read_csv('/Users/chineme/Downloads/Customer_Data.csv')

# Assuming your DataFrame is named df and already loaded

# Create a flag for delinquency where payments are less than minimum payments
df['Delinquent'] = df['PAYMENTS'] < df['MINIMUM_PAYMENTS']

# Calculate the delinquency rate as a percentage of total accounts
delinquency_rate = df['Delinquent'].mean() * 100

# Calculate the utilization rate
df['Utilization_Rate'] = df['BALANCE'] / df['CREDIT_LIMIT']


# Check for missing values
# missing_values = df.isnull().sum()
# print(missing_values)



# missing_data_indicator = pd.DataFrame(df.isnull().astype(int), columns=['Missing'])

# # Display the new DataFrame
# print(missing_data_indicator)

# Fill missing CREDIT_LIMIT with mean
df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean(), inplace=True)

# Calculate the ratio of MINIMUM_PAYMENTS to CREDIT_LIMIT for each non-missing row
df['MIN_PAY_TO_LIMIT_RATIO'] = df['MINIMUM_PAYMENTS'] / df['CREDIT_LIMIT']

# Calculate the median ratio
median_ratio = df['MIN_PAY_TO_LIMIT_RATIO'].median()

# Fill missing MINIMUM_PAYMENTS based on the median ratio
df['MINIMUM_PAYMENTS'].fillna(df['CREDIT_LIMIT'] * median_ratio, inplace=True)

# Drop the temporary ratio column
df.drop('MIN_PAY_TO_LIMIT_RATIO', axis=1, inplace=True)


# Display all rows in the MINIMUM_PAYMENTS column
#print(df['MINIMUM_PAYMENTS'])

# Check for missing values in the DataFrame
# missing_data = df.isnull().sum()

# # Display columns with missing data
# print("Columns with missing data:")
# print(missing_data[missing_data > 0])

# print (df)

# Check for duplicate entries
# duplicate_entries = df.duplicated().sum()
# print(f"Number of duplicate entries: {duplicate_entries}")

# # Remove duplicate entries
# df.drop_duplicates(inplace=True)


# Compute descriptive statistics
# descriptive_stats = df.describe()
# print(descriptive_stats)


# Compute descriptive statistics
descriptive_stats = df.describe()

# Transpose the DataFrame
descriptive_stats_transposed = descriptive_stats.T

# Display the transposed DataFrame
# print(descriptive_stats_transposed)


# Plot histogram of a single variable

# plt.figure(figsize=(10, 6))
# sns.histplot(df['BALANCE'], bins=30, kde=True)
# plt.title('Distribution of Balance')
# plt.xlabel('Balance')
# plt.ylabel('Frequency')
# plt.show()
# plt.close()


# Plot box plot of a single variable

# plt.figure(figsize=(10, 6))
# sns.boxplot(x=df['PURCHASES'])
# plt.title('Box plot of Purchases')
# plt.xlabel('Purchases')
# plt.show()


# Plot scatter plot of two variables
# plt.figure(figsize=(10, 6))
# sns.regplot(x='PURCHASES', y='CREDIT_LIMIT', data=df, scatter_kws={'s':10}, line_kws={'color':'red'})
# plt.title('Relationship between Purchases and Credit Limit')
# plt.xlabel('Purchases')
# plt.ylabel('Credit Limit')
# plt.show()

# print(f"Delinquency Rate: {delinquency_rate:.2f}%")

# # View the first few rows to confirm the new column
# print(df[['PAYMENTS', 'MINIMUM_PAYMENTS', 'Delinquent']].head())

# Plot utilization Rate
# plt.figure(figsize=(10, 6))
# sns.histplot(df['Utilization_Rate'], bins=30, kde=True)
# plt.title('Distribution of Utilization Rate')
# plt.xlabel('Utilization Rate')
# plt.ylabel('Frequency')
# plt.show()


# Calculate correlation between Utilization Rate and Delinquency
# correlation_util_delinquency = df[['Utilization_Rate', 'Delinquent']].corr()

# print("Correlation between Utilization Rate and Delinquency:")
# print(correlation_util_delinquency)

# # Visualize the correlation matrix with Utilization Rate and Delinquency
# plt.figure(figsize=(6, 4))
# sns.heatmap(correlation_util_delinquency, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title("Correlation between Utilization Rate and Delinquency")
# plt.show()
# plt.close()

#  CALCULATING APR WITH THE PAYMENT AND BALANCE ON CC

# Approximate the effective monthly interest rate ?
# This is a rough approximation, assuming that payments roughly cover the interest accrued on the balance
df['Effective_Monthly_Rate'] = (df['PAYMENTS'] / df['BALANCE']).clip(upper=1)

# Annualize the monthly rate to get APR (Approximation)
df['Approximate_APR'] = df['Effective_Monthly_Rate'] * 12 * 100  # Converting to percentage

# Display the DataFrame with new features

print(df[['CUST_ID', 'BALANCE', 'PAYMENTS', 'Effective_Monthly_Rate', 'Approximate_APR']])

df['Approximate_APR'].fillna(df['Approximate_APR'].median(), inplace=True)


# Define feature and target variable
X = df[['BALANCE']]
y = df['Approximate_APR']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Display the coefficients
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

