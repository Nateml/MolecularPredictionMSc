import ijson
from pathlib import Path




path = Path("C:\\Users\\Tom\\Documents\\Planning\\MolecularPredictionMSc\\Data Processing\\MoNa\\MoNA-export-All_Spectra.json")
with open(path, "rb") as f:
    count = sum(1 for _ in ijson.items(f, "item"))
print(count)