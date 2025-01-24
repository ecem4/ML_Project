data normalization -> stanardscaler
#Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Final datasets
tap_task_data = pd.read_csv("TAP-Alertness.csv")
cerq_data = pd.read_csv("CERQ.csv")
cope_data = pd.read_csv("COPE.csv")
lot_r_data = pd.read_csv("LOT-R.csv")
PSQ_data = pd.read_csv("PSQ.csv")
NEO_FFI_data = pd.read_csv("NEO_FFI.csv")
stai_g_x2_data = pd.read_csv("STAI_G_X2.csv")

#Data we analyze
data_to_analyze = cerq_data[["CERQ_SelfBlame", "CERQ_Rumination", "CERQ_Catastrophizing"]]
data_to_analyze["COPE_SelfBlame"] = cope_data["COPE_SelfBlame"]
data_to_analyze[["LOT_Optimism", "LOT_Pessimism"]] = lot_r_data[["LOT_Optimism", "LOT_Pessimism"]]
data_to_analyze[["PSQ_Worries", "PSQ_Tension"]] = PSQ_data[["PSQ_Worries", "PSQ_Tension"]]
data_to_analyze["NEOFFI_Neuroticism"] = NEO_FFI_data["NEOFFI_Neuroticism"]
data_to_analyze["NEOFFI_Extraversion"] = NEO_FFI_data["NEOFFI_Extraversion"]
data_to_analyze["STAI_Trait_Anxiety"] = stai_g_x2_data["STAI_Trait_Anxiety"]
data_to_analyze.head()

#clean data - I don't know how to do it

#Normalization
scaler = StandardScaler()
normalized_data = data_to_analyze.copy()
scaler.fit(normalized_data)
normalized_data.head()

#Correlation Matrix
correlation_matrix = normalized_data.corr()
correlation_matrix.head()

#Plot
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
plt.title("Correlation Matrix Heatmap")
plt.show()


