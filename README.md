# MolecularPredictionMSc

## Installation

Activate the virtualenv, then install the project as an editable package:

```bash
source env/bin/activate
pip install -e .
```

This exposes the source packages (`neural_net`, `predicates`, `data`, `config`) directly, so imports work from anywhere without a `src.` prefix:

```python
from neural_net.model import MultitaskMLP
from predicates.interface import PredictionInterface
from config import config
```
