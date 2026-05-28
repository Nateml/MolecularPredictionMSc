from pathlib import Path
p=Path('data/MoNA-filtered-curated-with-maccs-and-structure-deduped.json')
with p.open('rb') as f:
    head=f.read(400)
print(head.decode('utf-8','replace'))
