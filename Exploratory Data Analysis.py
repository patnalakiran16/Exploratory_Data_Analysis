import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Creating a sample student dataset
data = {
    "Student": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"],

    "Study_Hours": [2, 3, 4, 5, 6, 7, 8, 9, 1, 2,
                    3, 4, 5, 6, 7, 8, 9, 10, 4, np.nan],

    "Attendance": [60, 65, 70, 75, 80, 85, 90, 95, 50, 55,
                   65, 70, 75, 80, 85, 90, 95, 98, np.nan, 88],

    "Previous_Score": [45, 50, 55, 60, 65, 70, 75, 80, 40, 45,
                       50, 55, 60, 65, 70, 75, 80, 85, 58, 68],

    "Final_Score": [40, 45, 50, 55, 60, 65, 70, 75, 35, 40,
                    45, 50, 55, 60, 65, 70, 80, 90, 52, 72]
}

df = pd.DataFrame(data)

# Displaying the student data
print("Student Dataset:")
print(df)

# Checking the dataset details
print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

# Checking missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Checking duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Removing duplicate records
df = df.drop_duplicates()

# Replacing missing values with the average value
columns = ["Study_Hours", "Attendance",
           "Previous_Score", "Final_Score"]

df[columns] = df[columns].fillna(df[columns].mean())

print("\nCleaned Student Dataset:")
print(df)

# Performing basic statistical analysis
print("\nAverage Study Hours:")
print(round(df["Study_Hours"].mean(), 2))

print("\nAverage Final Score:")
print(round(df["Final_Score"].mean(), 2))

print("\nHighest Final Score:")
print(df["Final_Score"].max())

print("\nLowest Final Score:")
print(df["Final_Score"].min())

# Creating a histogram to show score distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Final_Score"], bins=6, edgecolor="black")
plt.title("Distribution of Student Final Scores")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.show()

# Comparing study hours and final scores
plt.figure(figsize=(8, 5))
plt.scatter(df["Study_Hours"], df["Final_Score"])
plt.title("Study Hours vs Final Scores")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.show()

# Showing marks scored by each student
plt.figure(figsize=(10, 5))
plt.bar(df["Student"], df["Final_Score"])
plt.title("Student Performance Analysis")
plt.xlabel("Student")
plt.ylabel("Final Score")
plt.show()

# Understanding score distribution using box plot
plt.figure(figsize=(8, 5))
plt.boxplot(df["Final_Score"])
plt.title("Student Score Distribution")
plt.ylabel("Final Score")
plt.show()

# Finding relationships between numerical columns
correlation = df[columns].corr()

plt.figure(figsize=(8, 5))
plt.imshow(correlation, cmap="coolwarm")
plt.colorbar()

plt.xticks(range(len(columns)), columns, rotation=45)
plt.yticks(range(len(columns)), columns)

plt.title("Correlation Between Student Performance Factors")
plt.tight_layout()
plt.show()

# Printing important findings from the dataset
print("\nImportant Observations:")

print("1. Average study hours:",
      round(df["Study_Hours"].mean(), 2))

print("2. Average final score:",
      round(df["Final_Score"].mean(), 2))

print("3. Highest score obtained:",
      df["Final_Score"].max())

print("4. Lowest score obtained:",
      df["Final_Score"].min())

study_correlation = df["Study_Hours"].corr(df["Final_Score"])

print("5. Relationship between study hours and final scores:",
      round(study_correlation, 2))

# Saving the cleaned data into a CSV file
df.to_csv("cleaned_student_data.csv", index=False)

print("\nCleaned dataset saved successfully.")
print("Exploratory Data Analysis completed successfully!")
