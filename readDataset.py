import pandas as pd
import os
# Load the CSV file into a pandas DataFrame
# Replace 'your_file.csv' with the path to your CSV file


class ReadData:
    
    @staticmethod
    def read(path):
        print(os.path.exists(path))
        df = pd.read_csv(path,sep="\t")
        
        # Display the first 5 rows of the dataset
        print("First 5 rows of the dataset:")
        print(df.head())

        print("\n-----------------------------\n")

        # Display basic information about the dataset (columns, data types, non-null values)
        print("Dataset info:")
        print(df.info())

        print("\n-----------------------------\n")

        # Display statistical summary for numerical columns
        print("Statistical summary (numerical columns):")
        print(df.describe())

        print("\n-----------------------------\n")

        # Display summary for categorical (object) columns
        print("Summary of categorical columns:")
        print(df.describe(include=['object']))

        print("\n-----------------------------\n")

        # Display number of missing values in each column
        print("Missing values per column:")
        print(df.isnull().sum())

        print("\n-----------------------------\n")

        # Display the shape of the dataset (rows, columns)
        print(f"Dataset shape: {df.shape[0]} rows and {df.shape[1]} columns")
        
# This condition checks if the script is being run directly
# (not imported as a module in another script)
if __name__ == "__main__":
    
    # Call the read method from ReadData class/module
    # Replace the path with the correct location of your CSV file
    #ReadData.read("datas/proteinGroups.csv")
    ReadData.read("datas/Phospho__STY_Sites.csv")