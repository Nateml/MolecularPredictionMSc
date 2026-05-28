# MolecularPredictionMSc

## Installation

Activate the virtualenv (or create one first), then install the project as an editable package:

```bash
source env/bin/activate
pip install -e .
```

This exposes the source packages (`asp`, `data`, `model`, `predicates`, `test`) directly, so imports work from anywhere without a `src.` prefix:

```python
from neural_net.model import MultitaskMLP
from predicates.interface import PredictionInterface
from config import config
```
