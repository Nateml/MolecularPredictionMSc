"""
Validation and Testing Script for DreaMS Encoder Integration

Run this script to verify that the DreaMS encoder is properly integrated
and working correctly with your multitask model.
"""

import torch
import torch.nn as nn
from loguru import logger

# Configure logging
logger.add("validation.log")


def test_dreams_encoder():
    """Test the DreaMS encoder standalone."""
    logger.info("=" * 80)
    logger.info("Testing DreaMS Encoder")
    logger.info("=" * 80)
    
    from src.model.dreams_encoder import DreaMSEncoder, spectrum_string_to_peaks, pad_peak_batch
    
    # Create encoder
    encoder = DreaMSEncoder(
        hidden_dim=256,
        num_transformer_layers=2,
        num_attention_heads=8,
    )
    logger.info(f"Created encoder: {encoder}")
    
    # Test with dummy peak data
    batch_size = 8
    max_peaks = 32
    
    # Create random peak data
    peaks = torch.randn(batch_size, max_peaks, 2).abs()
    peaks[:, :, 0] = peaks[:, :, 0] * 1000  # m/z: 0-1000
    peaks[:, :, 1] = peaks[:, :, 1] * 100   # intensity: 0-100
    
    # Create attention mask (all valid for this test)
    attention_mask = torch.ones(batch_size, max_peaks, dtype=torch.bool)
    
    # Forward pass
    output = encoder(peaks, attention_mask=attention_mask)
    
    assert output.shape == (batch_size, 256), f"Expected shape (8, 256), got {output.shape}"
    logger.info(f"✓ Encoder output shape correct: {output.shape}")
    
    # Test with precursor m/z
    precursor_mz = torch.randn(batch_size).abs() * 500
    output_with_precursor = encoder(peaks, precursor_mz=precursor_mz, attention_mask=attention_mask)
    assert output_with_precursor.shape == (batch_size, 256)
    logger.info(f"✓ With precursor m/z: {output_with_precursor.shape}")
    
    # Test spectrum string conversion
    spectrum_str = "100.5:500 200.3:300 300.1:400 400.2:200"
    peaks_from_str, precursor, mask = spectrum_string_to_peaks(spectrum_str)
    logger.info(f"✓ Spectrum conversion: {peaks_from_str.shape[0]} peaks extracted")
    
    logger.info("✓ All DreaMS encoder tests passed!\n")
    return True


def test_multitask_cnn_dreams():
    """Test the MultitaskCNNWithDreams model."""
    logger.info("=" * 80)
    logger.info("Testing MultitaskCNNWithDreams Model")
    logger.info("=" * 80)
    
    from src.model.model import MultitaskCNNWithDreams, categorical_sizes_from_df
    
    # Create dummy categorical sizes
    categorical_sizes = {
        "C_count": 10,
        "H_count": 20,
        "O_count": 8,
        "N_count": 5,
    }
    
    # Create model
    model = MultitaskCNNWithDreams(
        spectrum_len=1000,
        num_binary=166,
        categorical_sizes=categorical_sizes,
        num_regression=2,
        hidden_dim=128,
        num_transformer_layers=2,
    )
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test with dense spectrum input
    batch_size = 4
    spectrum_dense = torch.randn(batch_size, 1000).abs()
    
    output = model(spectrum_dense)
    
    assert "binary" in output
    assert "categorical" in output
    assert "quantile" in output
    
    assert output["binary"].shape == (batch_size, 166)
    assert len(output["categorical"]) == len(categorical_sizes)
    assert output["categorical"][0].shape == (batch_size, 10)
    
    logger.info(f"✓ Binary output: {output['binary'].shape}")
    logger.info(f"✓ Categorical outputs: {len(output['categorical'])} heads")
    logger.info(f"✓ Quantile outputs: {len(output['quantile'])} heads")
    
    # Test with peak list input
    from src.model.dreams_encoder import pad_peak_batch
    
    peak_lists = [
        torch.randn(10, 2).abs() for _ in range(batch_size)
    ]
    peaks_padded, attention_mask = pad_peak_batch(peak_lists, max_peaks=32)
    
    output_peaks = model(
        peaks_padded,
        attention_mask=attention_mask
    )
    
    assert output_peaks["binary"].shape == (batch_size, 166)
    logger.info(f"✓ Peak list input works: {output_peaks['binary'].shape}")
    
    # Test with precursor m/z
    precursor = torch.randn(batch_size).abs() * 500
    output_with_precursor = model(
        peaks_padded,
        precursor_mz=precursor,
        attention_mask=attention_mask
    )
    logger.info(f"✓ With precursor m/z: {output_with_precursor['binary'].shape}")
    
    # Test encoder freezing
    model.freeze_encoder()
    frozen_params = sum(1 for p in model.encoder.parameters() if not p.requires_grad)
    logger.info(f"✓ Freeze encoder: {frozen_params} frozen parameters")
    
    model.unfreeze_encoder()
    unfrozen_params = sum(1 for p in model.encoder.parameters() if p.requires_grad)
    logger.info(f"✓ Unfreeze encoder: {unfrozen_params} unfrozen parameters")
    
    logger.info("✓ All MultitaskCNNWithDreams tests passed!\n")
    return True


def test_weight_management():
    """Test weight saving and loading."""
    logger.info("=" * 80)
    logger.info("Testing Weight Management")
    logger.info("=" * 80)
    
    from src.model.dreams_weights import save_trained_encoder, load_saved_encoder
    from src.model.dreams_encoder import DreaMSEncoder
    from pathlib import Path
    import tempfile
    
    # Create encoder
    encoder = DreaMSEncoder(hidden_dim=128, num_transformer_layers=2)
    
    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_encoder.pt"
        
        save_trained_encoder(
            encoder,
            save_path=save_path,
            training_info={"test": "dummy", "epochs": 5}
        )
        logger.info(f"✓ Saved encoder to {save_path}")
        
        # Load
        loaded_encoder = load_saved_encoder(save_path, device="cpu")
        logger.info(f"✓ Loaded encoder from {save_path}")
        
        # Verify weights match
        for param1, param2 in zip(encoder.parameters(), loaded_encoder.parameters()):
            assert torch.allclose(param1, param2), "Loaded weights don't match saved weights"
        
        logger.info(f"✓ Weights verified to match")
    
    logger.info("✓ All weight management tests passed!\n")
    return True


def test_spectrum_conversion():
    """Test spectrum string to peak list conversion."""
    logger.info("=" * 80)
    logger.info("Testing Spectrum Conversion")
    logger.info("=" * 80)
    
    from src.model.dreams_encoder import spectrum_string_to_peaks, pad_peak_batch
    
    # Test various spectrum formats
    spectra = [
        "10.5:100 20.3:50 30.1:75",
        "15.2:120 25.4:80",
        "",  # Empty spectrum
        "5.1:10 10.2:20 15.3:30 20.4:40 25.5:50",
    ]
    
    peak_lists = []
    for spec in spectra:
        peaks, precursor, mask = spectrum_string_to_peaks(spec, mz_min=0, mz_max=1000)
        peak_lists.append(peaks)
        logger.info(f"  Spectrum: '{spec}' → {peaks.shape[0]} peaks")
    
    # Pad to same length
    padded, attention_mask = pad_peak_batch(peak_lists, max_peaks=10)
    
    assert padded.shape == (len(spectra), 10, 2)
    assert attention_mask.shape == (len(spectra), 10)
    
    logger.info(f"✓ Padded shape: {padded.shape}")
    logger.info(f"✓ Attention mask shape: {attention_mask.shape}")
    logger.info(f"✓ Valid peaks per spectrum: {attention_mask.sum(dim=1).tolist()}")
    
    logger.info("✓ All spectrum conversion tests passed!\n")
    return True


def test_compatibility():
    """Test backward compatibility with existing code."""
    logger.info("=" * 80)
    logger.info("Testing Backward Compatibility")
    logger.info("=" * 80)
    
    from src.model.model import MultitaskCNN, MultitaskCNNWithDreams
    
    categorical_sizes = {
        "feature_1": 5,
        "feature_2": 10,
    }
    
    # Create both models
    cnn_old = MultitaskCNN(
        spectrum_len=1000,
        num_binary=166,
        categorical_sizes=categorical_sizes,
    )
    
    cnn_new = MultitaskCNNWithDreams(
        spectrum_len=1000,
        num_binary=166,
        categorical_sizes=categorical_sizes,
    )
    
    # Test with same input
    batch_size = 2
    spectrum = torch.randn(batch_size, 1000).abs()
    
    output_old = cnn_old(spectrum)
    output_new = cnn_new(spectrum)
    
    # Check output structure
    assert set(output_old.keys()) == set(output_new.keys()) or \
           all(k in output_new for k in output_old.keys()), \
           "Output structure mismatch"
    
    # Check dimensions
    assert output_old["binary"].shape == output_new["binary"].shape
    assert len(output_old["categorical"]) == len(output_new["categorical"])
    
    logger.info(f"✓ Old model output: binary {output_old['binary'].shape}, "
                f"{len(output_old['categorical'])} categorical")
    logger.info(f"✓ New model output: binary {output_new['binary'].shape}, "
                f"{len(output_new['categorical'])} categorical")
    logger.info(f"✓ Backward compatible: Same input/output interface")
    
    logger.info("✓ All compatibility tests passed!\n")
    return True


def run_all_tests():
    """Run all validation tests."""
    logger.info("\n" + "=" * 80)
    logger.info("DREAMS ENCODER VALIDATION TEST SUITE")
    logger.info("=" * 80 + "\n")
    
    all_passed = True
    
    try:
        test_dreams_encoder()
    except Exception as e:
        logger.error(f"✗ DreaMS encoder test failed: {e}")
        all_passed = False
    
    try:
        test_multitask_cnn_dreams()
    except Exception as e:
        logger.error(f"✗ MultitaskCNNWithDreams test failed: {e}")
        all_passed = False
    
    try:
        test_weight_management()
    except Exception as e:
        logger.error(f"✗ Weight management test failed: {e}")
        all_passed = False
    
    try:
        test_spectrum_conversion()
    except Exception as e:
        logger.error(f"✗ Spectrum conversion test failed: {e}")
        all_passed = False
    
    try:
        test_compatibility()
    except Exception as e:
        logger.error(f"✗ Compatibility test failed: {e}")
        all_passed = False
    
    logger.info("=" * 80)
    if all_passed:
        logger.info("✓ ALL TESTS PASSED!")
    else:
        logger.error("✗ SOME TESTS FAILED - See errors above")
    logger.info("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
