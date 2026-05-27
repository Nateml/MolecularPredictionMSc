# CNN Flow — MultitaskCNN

Purpose
- Describe the data flow and components of `MultitaskCNN` implemented in `src/model/model.py`.

Overview
- Input: a single-channel spectrum vector of shape (batch, spectrum_len).
- Backbone: a stack of 1D convolutional blocks (Conv1d → BatchNorm1d → ReLU → Dropout) with interleaved MaxPool1d for downsampling.
- Pooling: global average pooling to collapse the sequence dimension to a fixed-size vector.
- Projection: a small MLP projects pooled channel features to a `global_feat_dim` feature vector.
- Heads: one linear `binary_head` and one `categorical_heads` per categorical target.

Forward pass (step-by-step)
1. Receive `x` shaped `(batch, spectrum_len)`.
2. If `x.dim() == 2`, do `x = x.unsqueeze(1)` → `(batch, 1, spectrum_len)`.
3. Pass through `conv_backbone` (several conv blocks + pooling): output `(batch, C_last, L_downsampled)`.
4. `global_pool` (AdaptiveAvgPool1d(1)) → `(batch, C_last, 1)`.
5. `proj` (Flatten + Linear + ReLU + Dropout) → `(batch, global_feat_dim)` feature vector.
6. Heads:
   - `binary_head(feat)` → `(batch, num_binary)`, raw logits for binary targets.
   - `[head(feat) for head in categorical_heads]` → list of `(batch, num_classes)` logits for each categorical target.

Shapes example
- spectrum_len = 1024, conv_channels = [32, 64, 128], global_feat_dim = 128
- Input: `(16, 1024)` → after unsqueeze `(16, 1, 1024)`
- After conv+pool x3 (each pool halves length): `(16, 128, 128)`
- After global_pool: `(16, 128, 1)` → flattened to `(16, 128)` → projected to `(16, 128)` feature vector

Integration notes
- No dataset shape changes required: training dataset may continue to yield `(batch, spectrum_len)` and the model will add the channel dim internally.
- If dataset already yields `(batch, 1, spectrum_len)`, `MultitaskCNN` still accepts it.
- Use the provided `src/model/run_training_cnn.py` to train the CNN; it mirrors the existing training flow and writes checkpoints to the same splits-based layout.

Hyperparameter tips
- `conv_channels`: [32,64,128] is a good starting point for medium-length spectra.
- `kernel_size`: 3 or 5 capture local patterns; larger kernels capture wider contexts.
- `pooling`: MaxPool1d(kernel_size=2) reduces sequence length quickly; adjust depth to avoid collapsing important detail.
- `global_feat_dim`: controls capacity for heads; 64–256 is a reasonable range.
- Regularisation: `Dropout` in conv blocks and `BatchNorm1d` help stability.

Debugging and sanity checks
- Run a single-batch forward pass and assert shapes of outputs.
- Visualise activations of early conv layers to ensure non-zero responses.
- If loss does not decrease, try reducing learning rate or simplifying the network (fewer channels).

When to use CNN vs MLP
- CNNs excel when local spectral patterns (peaks, fragment neighborhoods) matter.
- MLPs are simpler and may perform similarly if relevant information is global or if dataset is small.

Contact
- See `src/model/model.py` for implementation details and `src/model/run_training_cnn.py` to run training.
