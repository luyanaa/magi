#!/usr/bin/env python3
"""
Test Magi v2 integration with Brain MoE-PINN v2.
"""

import os
import sys
import torch
import warnings
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_magi_v2_import():
    """Test Magi v2 module imports."""
    print("Testing Magi v2 imports...")
    
    try:
        # Test 1: Magi v2 modules
        from brain_moe_pinn.magi.magi_v2 import (
            RotaryPositionEmbedding,
            GeGLU,
            FactorizedAttentionV2,
            MagiV2TransformerLayer,
            MagiV2TransformerEncoder,
            MagiV2EEGEncoder,
        )
        print("✓ Magi v2 modules imported successfully")
        
        # Test 2: Create instances
        rope = RotaryPositionEmbedding(dim=64)
        geglu = GeGLU(dim_in=1024, dim_out=4096)
        attn = FactorizedAttentionV2(hidden_dim=1024, num_heads=16)
        layer = MagiV2TransformerLayer(hidden_dim=1024, num_heads=16)
        encoder = MagiV2TransformerEncoder(num_layers=24, hidden_dim=1024)
        magi_encoder = MagiV2EEGEncoder(hidden_dim=1024, num_layers=24)
        
        print("✓ Magi v2 instances created")
        
        # Test 3: Forward pass
        B, C, T = 2, 64, 1024
        dummy_eeg = torch.randn(B, C, T)
        dummy_names = [[f'Ch{i}' for i in range(C)] for _ in range(B)]
        dummy_types = torch.zeros(B, C, dtype=torch.long)  # All scalp EEG
        
        with torch.no_grad():
            last_hidden, pooler = magi_encoder(
                eeg=dummy_eeg,
                channel_names=dummy_names,
                channel_types=dummy_types,
            )
        
        print(f"✓ Magi v2 forward pass: input {dummy_eeg.shape}, output {last_hidden.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Magi v2 import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_eeg_encoder_wrapper_v2():
    """Test EEGEncoderWrapperV2."""
    print("\nTesting EEGEncoderWrapperV2...")
    
    try:
        from brain_moe_pinn.encoders.eeg_encoder_v2 import EEGEncoderWrapperV2
        
        # Create wrapper
        wrapper = EEGEncoderWrapperV2(
            hidden_dim=1024,
            output_dim=2048,
            num_layers=24,
            freeze_encoder=False,  # For testing
            use_biot_embedding=True,
            use_channel_type_embed=True,
            max_channels=256,
            ecog_amplitude_scale=20.0,
            use_mamba2=False,
        )
        
        print("✓ EEGEncoderWrapperV2 created")
        
        # Test forward pass
        B, C, T = 2, 64, 1024
        dummy_eeg = torch.randn(B, C, T)
        dummy_names = [[f'Ch{i}' for i in range(C)] for _ in range(B)]
        dummy_types = torch.ones(B, C, dtype=torch.long)  # All ECoG
        
        with torch.no_grad():
            embeddings, pooler = wrapper(
                eeg=dummy_eeg,
                channel_names=dummy_names,
                channel_types=dummy_types,
                return_pooler=True,
            )
        
        print(f"✓ EEGEncoderWrapperV2 forward: input {dummy_eeg.shape}, embeddings {embeddings.shape}")
        
        # Test parameter counts
        params = wrapper.get_num_params()
        print(f"✓ Parameter counts: total={params.get('total', 0):,}, trainable={params.get('trainable', 0):,}")
        
        # Test context length
        wrapper.set_context_length(512)
        print("✓ Context length setting")
        
        # Test training step
        wrapper.set_training_step(epoch=0, total_epochs=10)
        print("✓ Training step update")
        
        return True
        
    except Exception as e:
        print(f"✗ EEGEncoderWrapperV2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_brain_moe_pinn_v2():
    """Test BrainMoEPINNV2."""
    print("\nTesting BrainMoEPINNV2...")
    
    try:
        from brain_moe_pinn.brain_moe_pinn_v2 import BrainMoEPINNV2, BrainMoEPINNConfig
        
        # Create config
        config = BrainMoEPINNConfig(
            use_magi_v2=True,
            eeg_hidden_dim=1024,
            eeg_num_layers=24,
            latent_dim=2048,
            max_channels=256,
            ecog_amplitude_scale=20.0,
            use_channel_type_embed=True,
            moe_num_experts=4,
            moe_top_k=2,
            freeze_encoders_epochs=1,
            use_mamba2=False,
            use_kda_decoder=True,
            use_active_inference=True,
            use_neurostorm=True,
        )
        
        print("✓ BrainMoEPINNConfig created")
        
        # Create model
        model = BrainMoEPINNV2(
            config=config,
            eeg_channels=64,  # Simulating ECoG
        )
        
        print("✓ BrainMoEPINNV2 created")
        
        # Test forward pass
        B, C, T = 2, 64, 1024
        R = 400  # fMRI regions
        
        dummy_eeg = torch.randn(B, C, T)
        dummy_fmri = torch.randn(B, R, T)  # ROI time series
        
        # Mixed channel types
        channel_types = torch.zeros(B, C, dtype=torch.long)
        channel_types[:, :32] = 0  # First 32: scalp EEG
        channel_types[:, 32:] = 1  # Last 32: ECoG
        
        channel_names = [[f'Ch{i}' for i in range(C)] for _ in range(B)]
        
        with torch.no_grad():
            output = model(
                eeg=dummy_eeg,
                fmri=dummy_fmri,
                channel_names=channel_names,
                channel_types=channel_types,
                mode='perception',
                num_steps=3,
                return_all=False,
            )
        
        print(f"✓ BrainMoEPINNV2 forward: EEG {dummy_eeg.shape}, fMRI {dummy_fmri.shape}")
        print(f"  Output keys: {list(output.keys())}")
        print(f"  z_final shape: {output['z_final'].shape}")
        print(f"  EEG recon shape: {output['eeg_recon'].shape}")
        print(f"  fMRI recon shape: {output['fmri_recon'].shape}")
        
        # Test parameter counts
        params = model.get_num_params()
        print(f"✓ Parameter counts: total={params.get('total', 0):,}, trainable={params.get('trainable', 0):,}")
        
        # Test training step
        model.set_training_step(epoch=0, total_epochs=10)
        print("✓ Training step update")
        
        # Test context length
        model.set_context_length(512)
        print("✓ Context length update")
        
        return True
        
    except Exception as e:
        print(f"✗ BrainMoEPINNV2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ecog_dataset():
    """Test ECoGDataset."""
    print("\nTesting ECoGDataset...")
    
    try:
        from brain_moe_pinn.utils.ecog_dataset import ECoGDataset
        
        # Create a temporary test directory
        import tempfile
        import shutil
        import numpy as np
        
        temp_dir = tempfile.mkdtemp()
        print(f"Creating test data in: {temp_dir}")
        
        try:
            # Create dummy ECoG data
            C, T = 64, 5120  # 64 channels, 10s at 512Hz
            dummy_data = np.random.randn(C, T).astype(np.float32)
            
            # Save as numpy file
            test_file = Path(temp_dir) / "test_ecog.npy"
            np.save(test_file, dummy_data)
            
            # Create dataset
            dataset = ECoGDataset(
                data_dir=temp_dir,
                sample_rate=512,
                seq_duration=5.0,
                patch_size=256,
                stride=128,
                max_channels=256,
                ecog_amplitude_scale=20.0,
                preload=True,
                use_mni_coords=False,
                require_mni=False,
            )
            
            print(f"✓ ECoGDataset created: {len(dataset)} samples")
            
            # Get a sample
            sample = dataset[0]
            print(f"✓ Sample loaded: keys={list(sample.keys())}")
            print(f"  EEG shape: {sample['eeg'].shape}")
            print(f"  Channel types: {sample['channel_types']}")
            
            # Test statistics
            stats = dataset.get_statistics()
            print(f"✓ Statistics: {stats}")
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
            print(f"Cleaned up test directory")
        
        return True
        
    except Exception as e:
        print(f"✗ ECoGDataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script_v2():
    """Test training script v2 imports and argument parsing."""
    print("\nTesting training script v2...")
    
    try:
        from brain_moe_pinn.scripts.train_v2 import parse_phases, create_model
        
        # Test phase parsing
        phases = parse_phases("-1,1,2,3")
        print(f"✓ Phase parsing: {len(phases)} phases")
        
        for i, phase in enumerate(phases):
            print(f"  Phase {i}: {phase.get('name', 'unknown')}")
        
        # Create dummy args for model creation test
        class DummyArgs:
            def __init__(self):
                self.eeg_channels = 64
                self.fmri_regions = 400
                self.latent_dim = 2048
                self.use_magi_v2 = True
                self.use_kda_decoder = True
                self.use_active_inference = True
                self.use_neurostorm = True
                self.use_mamba2 = False
                self.mamba2_kwargs = None
                self.eeg_checkpoint = None
                self.fmri_checkpoint = None
                self.max_channels = 256
                self.ecog_amplitude_scale = 20.0
        
        args = DummyArgs()
        
        # Test model creation (first phase)
        if phases:
            phase_config = phases[0]
            model, use_v2 = create_model(args, phase_config)
            print(f"✓ Model creation: use_v2={use_v2}, model_type={type(model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training script v2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Brain MoE-PINN v2 Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Magi v2 imports", test_magi_v2_import),
        ("EEGEncoderWrapperV2", test_eeg_encoder_wrapper_v2),
        ("BrainMoEPINNV2", test_brain_moe_pinn_v2),
        ("ECoGDataset", test_ecog_dataset),
        ("Training script v2", test_training_script_v2),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Test: {test_name}")
        print(f"{'='*40}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Test {test_name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:30} {status}")
        if not success:
            all_passed = False
    
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())