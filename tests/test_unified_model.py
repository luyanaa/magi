"""Regression checks for the unified species/model contract."""

import json
import pytest
from pathlib import Path

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]

from brain_moe_pinn import BrainMoEPINN, BrainMoEPINNConfig
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



def test_sde_species_profile_reaches_model_and_crps_pilot():
    sde_config = ExperimentConfig.from_file(
        ROOT / "configs/species/c_elegans_unified_sde.json")
    model = BrainMoEPINN(
        use_neurostorm=False, **sde_config.model_kwargs()).eval()

    assert model.transition_mode == "sde"
    assert model.diffusion_rank == 16
    assert model.transition_diffusion.factor_basis.shape == (192, 16)
    assert model.diffusion_floor == pytest.approx(1e-4)
    assert model.diffusion_scale == pytest.approx(1e-2)
    assert model.stochastic_samples == 8
    assert model.noise_mode == "off"
    assert model.scale_anchor == "off"
    adapter_config = BrainMoEPINNConfig.from_experiment(sde_config)
    assert adapter_config.scale_anchor == "off"

    with (ROOT / "configs/data/"
          "c_elegans_toyoshima_component_sde_pilot.json").open() as handle:
        pilot = json.load(handle)
    assert pilot["forecast_crps"] == 1.0
    assert pilot["forecast_center"] == 0.1
    assert pilot["forecast_recon_extra"] == {"calcium": 0.0}
    assert pilot["component_forecast"]["loss_weight"] == 0.0

    nearzero_config = ExperimentConfig.from_file(
        ROOT / "configs/species/c_elegans_unified_sde_nearzero.json")
    nearzero = BrainMoEPINNConfig.from_experiment(nearzero_config)
    assert nearzero.transition_mode == "sde"
    assert nearzero.diffusion_rank == 16
    assert nearzero.diffusion_floor == pytest.approx(1e-8)
    assert nearzero.diffusion_scale == 0.0
    assert nearzero.stochastic_samples == 8
    assert nearzero.scale_anchor == "off"

    with (ROOT / "configs/species/c_elegans_unified_sde.json").open() as handle:
        reference_profile = json.load(handle)
    with (ROOT / "configs/species/c_elegans_unified_sde_nearzero.json").open() as handle:
        nearzero_profile = json.load(handle)
    assert reference_profile["data"] == nearzero_profile["data"]
    assert reference_profile["training"] == nearzero_profile["training"]
    reference_features = dict(reference_profile["features"])
    nearzero_features = dict(nearzero_profile["features"])
    reference_features.pop("diffusion_floor")
    reference_features.pop("diffusion_scale")
    nearzero_features.pop("diffusion_floor")
    nearzero_features.pop("diffusion_scale")
    assert reference_features == nearzero_features


    with (ROOT / "configs/data/"
          "c_elegans_toyoshima_component_sde_diffusion_only.json").open() as handle:
        diffusion_only = json.load(handle)
    assert diffusion_only["forecast_objective"] == "crps_only"
    assert diffusion_only["forecast_crps"] == 1.0
    assert "component_forecast" not in diffusion_only
    with (ROOT / "configs/data/"
          "c_elegans_toyoshima_component_sde_pilot.json").open() as handle:
        baseline_data = json.load(handle)
    compared_data_keys = {
        "root", "modalities", "roles", "control_modalities", "seq_seconds",
        "batch_size", "num_workers", "pin_memory", "persistent_workers",
        "prefetch_factor", "split_by", "val_frac", "include_sources",
        "align_channels", "max_union_channels", "channel_order_file",
        "return_masks", "return_next_step_targets", "future_steps",
        "recon_loss_mix",
        "rollout_steps", "rollout_horizon_weights", "normalization",
        "control_width", "use_trials",
    }
    assert {
        key: baseline_data[key] for key in compared_data_keys
    } == {
        key: diffusion_only[key] for key in compared_data_keys
    }

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


