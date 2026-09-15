"""Unit checks for calibrated calcium-to-voltage reconstruction."""

import pytest


torch = pytest.importorskip("torch")

from brain_moe_pinn.core.celegans_physiology import (
    FluorescenceCalibration,
    calcium_from_voltage,
    capacitive_current_density,
    hill_fluorescence,
    invert_hill_fluorescence,
    reconstruct_voltage_current,
    voltage_from_calcium,
)


def test_hill_forward_inverse_preserves_calibrated_calcium():
    calcium = torch.tensor([25.0, 100.0, 235.0, 500.0])
    fluorescence = hill_fluorescence(
        calcium, f0=100.0, fmax=1100.0, kd_nM=235.0, hill_h=3.3
    )
    recovered, valid = invert_hill_fluorescence(
        fluorescence, f0=100.0, fmax=1100.0, kd_nM=235.0, hill_h=3.3
    )
    assert valid.all()
    assert torch.allclose(recovered, calcium, atol=2e-4, rtol=2e-5)


def test_hill_inverse_masks_out_of_operating_range_instead_of_clipping():
    recovered, valid = invert_hill_fluorescence(
        torch.tensor([99.0, 100.0, 600.0, 1100.0, 1101.0]),
        f0=100.0,
        fmax=1100.0,
        kd_nM=235.0,
        hill_h=3.3,
    )
    assert valid.tolist() == [False, True, True, True, False]
    assert torch.isnan(recovered[[0, 4]]).all()
    assert torch.isfinite(recovered[1:4]).all()


def test_gaussian_voltage_inversion_requires_branch():
    voltage = torch.tensor([-2.0, 0.0, 2.0])
    calcium = calcium_from_voltage(
        voltage,
        model="gaussian",
        c0_nM_per_mV=100.0,
        c1_nM=10.0,
        mu_mV=0.0,
        sigma_mV=2.0,
    )
    with pytest.raises(ValueError, match="non-unique"):
        voltage_from_calcium(
            calcium,
            model="gaussian",
            c0_nM_per_mV=100.0,
            c1_nM=10.0,
            mu_mV=0.0,
            sigma_mV=2.0,
        )
    ascending, ascending_valid = voltage_from_calcium(
        calcium[:2],
        model="gaussian",
        c0_nM_per_mV=100.0,
        c1_nM=10.0,
        mu_mV=0.0,
        sigma_mV=2.0,
        branch="ascending",
    )
    descending, descending_valid = voltage_from_calcium(
        calcium[1:],
        model="gaussian",
        c0_nM_per_mV=100.0,
        c1_nM=10.0,
        mu_mV=0.0,
        sigma_mV=2.0,
        branch="descending",
    )
    assert ascending_valid.all() and descending_valid.all()
    assert torch.allclose(ascending, voltage[:2], atol=2e-5, rtol=2e-5)
    assert torch.allclose(descending, voltage[1:], atol=2e-5, rtol=2e-5)


def test_capacitive_current_density_uses_millivolt_to_volt_conversion():
    current, valid = capacitive_current_density(
        torch.tensor([0.0, 10.0, 20.0]),
        dt_s=0.1,
        capacitance_uF_per_cm2=38.5,
    )
    expected = 38.5e-6 * (10.0e-3 / 0.1)
    assert not valid[0]
    assert valid[1:].all()
    assert torch.isnan(current[0])
    assert torch.allclose(current[1:], torch.full((2,), expected))


def test_calibrated_reconstruction_reports_physical_units_and_masks():
    calibration = FluorescenceCalibration(
        f0=100.0,
        fmax=1100.0,
        hill_kd_nM=235.0,
        hill_h=3.3,
        voltage_model="linear",
        c0_nM_per_mV=2.0,
        c1_nM=50.0,
    )
    voltage = torch.tensor([0.0, 1.0, 2.0])
    fluorescence = hill_fluorescence(
        calcium_from_voltage(
            voltage, c0_nM_per_mV=2.0, c1_nM=50.0
        ),
        f0=100.0,
        fmax=1100.0,
        kd_nM=235.0,
        hill_h=3.3,
    )
    output = reconstruct_voltage_current(fluorescence, calibration, dt_s=0.5)
    assert output["fluorescence_valid"].all()
    assert output["voltage_valid"].all()
    assert torch.allclose(output["voltage_mV"], voltage, atol=2e-4, rtol=2e-4)
    assert not output["current_valid"][0]
    assert output["current_valid"][1:].all()
def test_standardized_yc260_keeps_one_latent_to_observed_filter_without_hill():
    from brain_moe_pinn.core.emission import SensorEmission
    from brain_moe_pinn.core.sensors import resolve_sensor

    spec = resolve_sensor(
        "calcium",
        {
            "imaging": "spinning_disk_4d",
            "reporter": "yc2.60",
            "readout": "ratio",
            "observed_space": "standardized",
            "already_filtered": True,
        },
    )
    emission = SensorEmission(max_channels=1, spec=spec, init_tau_s=1.0)
    signal = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]])
    output = emission(signal, dt=0.1)
    assert emission.apply_lowpass
    assert not emission.apply_hill
    assert output[0, 0, 1] < output[0, 0, 2] < output[0, 0, 3]
    summary = emission.parameter_summary(1)
    assert summary["observed_space"] == "standardized"
    assert summary["already_filtered"] is True
    assert summary["temporal_filter_applied"] is True
