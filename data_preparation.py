'''''Function nr 1 :  Data_preparation
Input: Dictionary where keys are chosen datasets and values are specific columns
Output: Dataset with chosen variables form input datasets which are normalized and lack of data are excluded. 
loop in the function will go through keys of input dict and values of each keys to import data; emerge them in one dataset. 
After that next loop will exclude lack of data and data will be normalized '''''

import pandas as pd
from sklearn.preprocessing import StandardScaler

def Data_preparation(datasets_dict):
    merged_df = pd.DataFrame()  # This is an empty dataframe to hold the merged data from all datasets
    
    # Iterate through the dictionary, read each dataset, select specified columns, and merge them
    for dataset, columns in datasets_dict.items():
        # Import dataset
        df = pd.read_csv(dataset)
        
        # Select specific columns
        df = df[columns].apply(pd.to_numeric, errors='coerce')
        # Merge with the main dataframe (concatenate along axis=1)
        merged_df = pd.concat([merged_df, df], axis=1)
    
    # Remove rows with missing values (NaN) to ensure the dataset is complete before further processing
    merged_df.dropna(inplace=True)   

    # Standardize (normalize) the data using StandardScaler
    scaler = StandardScaler()
    standardized_data = pd.DataFrame(scaler.fit_transform(merged_df), columns=merged_df.columns)
    
    return standardized_data


datasets_dict = { 'CERQ.csv': ['CERQ_SelfBlame', 'CERQ_Rumination', 'CERQ_Catastrophizing'],'COPE.csv': ['COPE_SelfBlame'],
'LOT-R.csv': ['LOT_Optimism', 'LOT_Pessimism'],
    'PSQ.csv': ['PSQ_Worries', 'PSQ_Tension'],
    'NEO_FFI.csv': ['NEOFFI_Neuroticism', 'NEOFFI_Extraversion'],
    'STAI_G_X2.csv': ['STAI_Trait_Anxiety']}

data_to_analyze = Data_preparation(datasets_dict)
print(data_to_analyze)

