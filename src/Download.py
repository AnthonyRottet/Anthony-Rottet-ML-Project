import kagglehub
from pathlib import Path
import shutil

# 1. Download dataset
path = Path(
    kagglehub.dataset_download("neurocipher/heartdisease")
)

# 2. Go up one level (..) to find the data folder in the root
Path("../data/raw").mkdir(parents=True, exist_ok=True)

# 3. Copy the file to the root data folder
shutil.copy(
    path / "Heart_Disease_Prediction.csv",
    "../data/raw/heart_disease.csv"
)

print("done", path)
