import kagglehub
from pathlib import Path
import shutil

# Download dataset
path = Path(
    kagglehub.dataset_download("neurocipher/heartdisease")
)

Path("../data/raw").mkdir(parents=True, exist_ok=True)

shutil.copy(
    path / "Heart_Disease_Prediction.csv",
    "../data/raw/heart_disease.csv"
)

print("done", path)
