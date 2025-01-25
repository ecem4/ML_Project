class DataPreprocessor:
        def __init__(self, datasets_dict):
        self.datasets_dict = datasets_dict
        self.merged_df = pd.DataFrame() #stores merged datasets
        
        def merge_data(self):
        # Load datasets, select columns, and merge them #
                for dataset, columns in self.datasets_dict.items():
                        df = pd.read_csv(dataset)  # Load the dataset
                        df = df[columns].apply(pd.to_numeric, errors='coerce')  # Convert columns to numeric
                        self.merged_df = pd.concat([self.merged_df, df], axis=1)  # Merge the datasets into one dataframe
        return self.merged_df
        
        def miss_val(self, strategy="drop"):
       ''' Handle missing values in the dataset according to the specified strategy:
            strategy (str): The method to handle missing values. Options are:
                            - "drop": Drop rows with any missing values - the chosen one 
                            - "mean": Fill missing values with the mean of each column
                            - "ffill": Forward fill missing values '''
                if strategy == "drop":
                        self.merged_df.dropna(inplace=True)
                elif strategy == "mean":
                        self.merged_df.fillna(self.merged_df.mean(), inplace=True)
                elif strategy == "ffill":
                        self.merged_df.fillna(method='ffill', inplace=True)
                else:
                        raise ValueError("Invalid strategy. Choose from "drop", "mean", or "ffill" ")

                


methods for: 
loading and merging datasets - done 
Checking Datatypes
encode categorical 
- One-Hot Encoding, - Label Encoding
handling missing values - done 
normalize data - done (minmax or standard scaler)

all of this at once also 

