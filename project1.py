import pandas as pd
tap_task_data = pd.read_csv("TAP-Alertness.csv")
data_to_analyze = tap_task_data[["ID", "TAP_A_5", "TAP_A_10"]]
data_to_analyze

data_to_analyze_copy = data_to_analyze

PSQ_data = pd.read_csv("PSQ.csv")
NEO_FFI_data = pd.read_csv("NEO_FFI.csv")

data_to_analyze_copy["PSQ_OverallScore"] = PSQ_data["PSQ_OverallScore"]
data_to_analyze_copy["NEOFFI_Neuroticism"] = NEO_FFI_data["NEOFFI_Neuroticism"]
data_to_analyze_copy["NEOFFI_Extraversion"] = NEO_FFI_data["NEOFFI_Extraversion"]

data_to_analyze_copy
