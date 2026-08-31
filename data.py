import pandas as pd

data = pd.read_csv("creditcard.csv")

print(data.head())
print(data.shape)
print(data.info())
print(data["Class"].value_counts())
print("Normal transactions:", (data["Class"] == 0).sum())
print("Fraud transactions:", (data["Class"] == 1).sum())
import matplotlib.pyplot as plt

data["Class"].value_counts().plot(kind="bar")

plt.title("Normal vs Fraud Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.show()
print(data.groupby("Class")["Amount"].mean())
print(data.isnull().sum().sum())
