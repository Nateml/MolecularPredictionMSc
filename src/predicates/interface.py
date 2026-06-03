from typing import Union, Dict, Optional
from pydantic import BaseModel, RootModel, field_validator, model_validator
from predicates.definitions import KEYS, FEATURE_VECTOR_DEFINITIONS_CONTINUOUS

KeyID = Union[int, str]
"""Identifier for a prediction entry, can be an integer or string. E.g. MACCS key ID 154 corresponds to the presence of a carbonyl group."""

Value = Union[int, str]
"""Value type for prediction sets, can be an integer or string."""

_CONTINUOUS_KEYS = frozenset(FEATURE_VECTOR_DEFINITIONS_CONTINUOUS.keys())


class PredictionEntry(BaseModel):
    """
    A single prediction entry containing a prediction set and associated metadata.

    Attributes:
        pred_set: Set of predicted values.
        alpha: Confidence level or significance threshold parameter that the pred_set was constructed with.
        proba: Optional probability distribution over the prediction set.
    """
    pred_set: list[Value]
    # Metadata:
    alpha: float = 0.05
    proba: Optional[list[float]] = None

    @field_validator('pred_set')
    @classmethod
    def non_empty(cls, v):
        if not v:
            raise ValueError("Prediction set cannot be empty")
        return v

    @model_validator(mode='after')
    def check_proba_alignment(self):
        if self.proba is not None and len(self.proba) != len(self.pred_set):
            raise ValueError("Length of probability distribution must match the length of the prediction set")
        return self


class ContinuousRange(BaseModel):
    """Predicted [lo, hi] interval for a continuous structural feature.

    Used as a value in PredictionInterface for features that are predicted via
    quantile regression rather than discrete bins. compile_asp_constraint converts
    these to ASP integrity constraints: :- pred(N), N < lo. :- pred(N), N > hi.
    """
    lo: float
    hi: float


class PredictionInterface(RootModel[dict[KeyID, Union[PredictionEntry, int, ContinuousRange]]]):
    """
    Interface for holding a collection of prediction entries, indexed by their KeyIDs.

    Must always include num_carbons, num_hydrogens, num_nitrogens, and num_oxygens as keys with integer values representing the counts of each atom type.

    Continuous structural features (e.g. alcohol, ring_count) may be included as
    ContinuousRange values; compile_asp_constraint will generate range constraints for them.

    Example usage:
    ```
    predictions = {
        154: PredictionEntry(pred_set=[1], alpha=0.05, proba=[0.98]),
        'num_carbons': 7,
        'num_hydrogens': 16,
        'num_nitrogens': 0,
        'num_oxygens': 0,
        'alcohol': ContinuousRange(lo=1.0, hi=3.0),
    }
    interface = PredictionInterface.model_validate(predictions)
    """

    @model_validator(mode='after')
    @classmethod
    def validate_keys(cls, data):
        allowed_keys = KEYS()
        input_keys = set(data.root.keys())

        invalid = input_keys - allowed_keys
        if invalid:
            raise ValueError(f"Invalid keys in prediction interface: {invalid}. See definitions.py for allowed keys.")

        required_atom_keys = {'num_carbons', 'num_hydrogens', 'num_nitrogens', 'num_oxygens'}
        missing = required_atom_keys - input_keys
        if missing:
            raise ValueError(f"Missing required atom count keys: {missing}. Must include num_carbons, num_hydrogens, num_nitrogens, and num_oxygens.")

        return data

    @model_validator(mode='after')
    @classmethod
    def validate_types(cls, data):
        required_atom_keys = {'num_carbons', 'num_hydrogens', 'num_nitrogens', 'num_oxygens', 'num_sulfurs', 'num_phosphorus'}

        for key in set(data.root.keys()):
            value = data.root[key]
            if key in required_atom_keys:
                if not isinstance(value, int):
                    raise ValueError(f"Atom count for {key} must be an integer, got {type(value)}")
            elif key in _CONTINUOUS_KEYS:
                if not isinstance(value, ContinuousRange):
                    raise ValueError(f"Continuous range for {key} must be a ContinuousRange instance, got {type(value)}")
            else:
                if not isinstance(value, PredictionEntry):
                    raise ValueError(f"Prediction entry for {key} must be a PredictionEntry instance, got {type(value)}")

        return data
