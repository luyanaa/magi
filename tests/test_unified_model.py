"""Regression checks for the unified species/model contract."""

import pytest
from pathlib import Path

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]

from brain_moe_pinn import BrainMoEPINN
from brain_moe_pinn.config import ExperimentConfig


def test_species_profiles_load_and_validate():
    expected_moe = {
        "c_elegans": False,
        "zebrafish": False,
        "mouse": True,
        "human": True,
    }
    expected_poisson_rank = {
        "c_elegans": 32,
        "zebrafish": 64,
        "mouse": 64,
        "human": 128,
    }
    for species in ("c_elegans", "zebrafish", "mouse", "human"):
        config = ExperimentConfig.from_file(ROOT / "configs" / "species" / f"{species}.json")
        assert config.species == species
        assert species in config.species_vocab
        assert config.training.ladder_stage == species
        assert config.data.modalities
        assert config.features.use_moe is expected_moe[species]
        assert config.features.poisson_rank == expected_poisson_rank[species]
        assert config.model_kwargs()["use_moe"] is expected_moe[species]
        assert config.model_kwargs()["poisson_rank"] == expected_poisson_rank[species]
        derived = ExperimentConfig.from_species(species)
        assert derived.latent_dim == {
            "c_elegans": 192,
            "zebrafish": 384,
            "mouse": 512,
            "human": 1024,
        }[species]
        assert derived.features.use_moe is expected_moe[species]
        assert derived.features.poisson_rank == expected_poisson_rank[species]


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


def test_no_moe_path_uses_only_structured_dynamics():
    model = BrainMoEPINN(
        latent_dim=8,
        use_neurostorm=False,
        use_kda_decoder=False,
        use_active_inference=False,
        generic_observation_only=True,
        use_moe=False,
        poisson_rank=4,
    ).eval()
    output = model.forward_modalities(
        {"calcium": torch.randn(1, 4, 16)},
        num_steps=1,
    )
    assert model.moe_velocity is None
    assert output["z_next"].shape == (1, 8)
    assert torch.isfinite(output["z_next"]).all()
    dynamics = model.get_dynamics_num_params()
    assert dynamics["moe_velocity"] == 0
    assert dynamics["total"] == dynamics["velocity_brain"]
    model.reset_router_state(batch_size=1)
 
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
        "get_dynamics_num_params",
    ):
        assert callable(getattr(model, name, None)), name
    model.set_training_step(epoch=0, total_epochs=2)
    model.set_context_length(512)
    assert model._current_context_length == 512
    assert model.get_num_params()["total"] > 0

def test_dynamics_parameter_diagnostics_exclude_observation_stack():
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
    dynamics = model.get_dynamics_num_params()
    assert dynamics["latent_dim"] == 8
    assert dynamics["total"] == (
        dynamics["velocity_brain"] + dynamics["moe_velocity"]
    )
    assert dynamics["total"] < model.get_num_params()["total"]


