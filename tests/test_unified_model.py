"""Regression checks for the unified species/model contract."""

import pytest
from pathlib import Path

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]

from brain_moe_pinn import BrainMoEPINN
from brain_moe_pinn.config import ExperimentConfig


def test_species_profiles_load_and_validate():
    for species in ("c_elegans", "zebrafish", "mouse", "human"):
        config = ExperimentConfig.from_file(ROOT / "configs" / "species" / f"{species}.json")
        assert config.species == species
        assert species in config.species_vocab
        assert config.training.ladder_stage == species
        assert config.data.modalities


def test_feature_gates_reject_incompatible_profiles():
    with pytest.raises(ValueError, match="paired_next_step_targets"):
        ExperimentConfig.from_dict({
            "species": "mouse",
            "features": {"use_active_inference": True},
            "data": {"modalities": ["calcium"], "paired_next_step_targets": False},
        })

    with pytest.raises(ValueError, match="use_meg"):
        ExperimentConfig.from_dict({
            "species": "mouse",
            "features": {"use_meg": True},
            "data": {"modalities": ["calcium"]},
        })


def test_forward_modalities_uses_shared_dynamics_and_species_metadata():
    torch.manual_seed(0)
    model = BrainMoEPINN(
        eeg_channels=2,
        fmri_regions=3,
        latent_dim=8,
        use_neurostorm=False,
        use_kda_decoder=False,
        use_active_inference=False,
        use_species_conditioning=True,
        species="c_elegans",
        species_vocab=["c_elegans", "human"],
        perturbation_dim=2,
        moe_num_shared=1,
        moe_num_routed=1,
        generic_observation_only=True,
        moe_top_k=1,
    ).eval()
    signals = {
        "calcium": torch.randn(2, 12, 32),
        "behavior": torch.randn(2, 3, 32),
    }
    output = model.forward_modalities(
        signals,
        species_names=["c_elegans", "human"],
        perturbation=torch.ones(2, 2),
        num_steps=2,
        return_all=True,
    )
    assert output["z_next"].shape == (2, 8)
    assert output["states"].shape == (2, 3, 8)
    assert set(output["modality_tokens"]) == {"calcium", "behavior"}
    assert torch.isfinite(output["z_next"]).all()
 
def test_canonical_model_public_api():
    model = BrainMoEPINN(
        latent_dim=8,
        use_neurostorm=False,
        use_kda_decoder=False,
        use_active_inference=False,
        generic_observation_only=True,
        moe_num_shared=1,
        moe_num_routed=1,
        moe_top_k=1,
    )
    for name in (
        "forward",
        "forward_modalities",
        "set_training_step",
        "set_context_length",
        "load_pretrained",
        "reset_history",
        "reset_router_state",
        "get_num_params",
    ):
        assert callable(getattr(model, name, None)), name
    model.set_training_step(epoch=0, total_epochs=2)
    model.set_context_length(512)
    assert model._current_context_length == 512
    assert model.get_num_params()["total"] > 0


