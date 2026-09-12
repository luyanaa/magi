"""Imaging methods and functional reporters: the observation channel registry.

A species ladder mixes corpora that differ in *how* the volume was acquired and
*what* the fluorescent protein reports. Both change what a signal means and what
a model may claim, so they belong in the profile next to the sampling rate
rather than in a comment:

* **Imaging method** fixes the frame integration window (a z-stack or a
  scanning pattern takes time, so a "frame" is an integral over
  ``volume_integration_s``), the achievable volume rate, and the *slice/plane
  phase smear* (neurons in different planes are not sampled simultaneously, so
  sub-volume lags are unidentifiable). Light-field (LFM/XLFM) is the exception:
  one snapshot yields the whole volume, so its integration is the camera
  exposure.
* **Reporter** fixes the readout kind (``dff`` / ``ratio`` / ``voltage`` /
  ``static``), the kinetics that low-pass the underlying state, the Hill
  parameters when the readout saturates, and whether dynamics may be modelled
  at all (pERK is a minutes-scale biochemical history in fixed tissue: no
  windowed next-step objective is meaningful).

Entries carry a ``source`` and a ``verified`` flag. Numbers without a
verifiable source stay ``None`` and are marked unverified rather than guessed;
``SensorSpec`` then reports ``calibration_required=True`` so a fitted emission
time constant is not mistaken for a literature value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "ImagingMethod",
    "Reporter",
    "SensorSpec",
    "IMAGING_METHODS",
    "REPORTERS",
    "READOUT_KINDS",
    "resolve_sensor",
    "sensor_spec_from_profile",
]

READOUT_KINDS = ("dff", "ratio", "absolute", "voltage", "static", "bold")


@dataclass(frozen=True)
class ImagingMethod:
    """How a volume was acquired and what that costs in time."""

    name: str
    label: str
    volume_integration_s: Optional[float]
    """Physical time one frame spans (z-stack or scan). ``None`` = unknown."""
    slice_phase_smear_s: Optional[float]
    """Max timing offset between neurons in one frame (scan/stack order)."""
    typical_rate_hz: Optional[tuple]
    resolution_note: str
    dynamics_note: str
    plane_interval_s: Optional[float] = None
    """Time between consecutive planes: the same-plane lag floor."""
    source: Optional[str] = None
    verified: bool = False


IMAGING_METHODS: Dict[str, ImagingMethod] = {
    "lsfm_spim": ImagingMethod(
        name="lsfm_spim",
        label="Light-sheet / selective-plane illumination (LSFM, SPIM)",
        volume_integration_s=0.914,
        slice_phase_smear_s=0.914,
        plane_interval_s=0.0127,
        typical_rate_hz=(0.5, 4.0),
        resolution_note="cellular, whole-brain; one plane per camera exposure",
        dynamics_note=(
            "volume rate is the binding constraint; planes are acquired "
            "sequentially (ZAPBench: 72 planes, 914 ms/volume, ~12.7 ms/plane), "
            "so same-plane lags resolve to ~12.7 ms while cross-plane timing is "
            "uncertain up to one volume"),
        source="ZAPBench, arXiv:2503.02618: 2048x1328x72x7879 voxels at "
               "406 nm x 406 nm x 4 um x 914 ms",
        verified=True,
    ),
    "lfm_xlfm": ImagingMethod(
        name="lfm_xlfm",
        label="Light-field / extended light-field (LFM, XLFM)",
        volume_integration_s=0.01,
        slice_phase_smear_s=0.0,
        plane_interval_s=0.0,
        typical_rate_hz=(10.0, 100.0),
        resolution_note="single snapshot -> full volume; resolution recovered "
                        "by deconvolution/PSF fitting",
        dynamics_note=(
            "all neurons share one exposure (no slice smear), so the frame "
            "integration is the camera exposure rather than a stack duration"),
        source="LFM/XLFM single-snapshot volumetric imaging",
        verified=False,
    ),
    "rs_lsfm": ImagingMethod(
        name="rs_lsfm",
        label="Remote-scanning LSFM",
        volume_integration_s=None,
        slice_phase_smear_s=None,
        plane_interval_s=None,
        typical_rate_hz=(1.0, 10.0),
        resolution_note="scanned sheet with remote focus; uneven illumination "
                        "across the field is a known artefact",
        dynamics_note="scan order sets the phase smear, as for LSFM",
        source="fill volume/plane timing from the acquisition metadata before "
               "trusting any lag-resolved metric",
        verified=False,
    ),
    "scape_3d_aod_2p": ImagingMethod(
        name="scape_3d_aod_2p",
        label="SCAPE / 3D-AOD two-photon",
        volume_integration_s=None,
        slice_phase_smear_s=None,
        plane_interval_s=None,
        typical_rate_hz=(0.5, 10.0),
        resolution_note="two-photon, deeper penetration; AOD/SCAPE scanning",
        dynamics_note=(
            "per-plane scanning: the volume period bounds resolvable lags and "
            "introduces a plane-order phase offset"),
        verified=False,
    ),
    "spinning_disk_4d": ImagingMethod(
        name="spinning_disk_4d",
        label="Spinning-disk confocal 4D (piezo z-scan)",
        volume_integration_s=0.245,
        slice_phase_smear_s=0.245,
        plane_interval_s=0.0111,
        typical_rate_hz=(3.0, 6.0),
        resolution_note="22 planes per volume in the salt corpus",
        dynamics_note=(
            "each ROI is integrated over the whole z-stack (~0.245 s/volume at "
            "the pilot's median 4.08 Hz); plane order smears cross-neuron "
            "timing by up to one volume"),
        source="Toyoshima et al. 2024 (PLOS Comput Biol 1011848), Methods: "
               "22 slices/volume, 6,000 volumes; stimulation_timing.xlsx gives "
               "the per-worm rate",
        verified=True,
    ),
    "widefield_2p": ImagingMethod(
        name="widefield_2p",
        label="Widefield / two-photon mesoscale",
        volume_integration_s=None,
        slice_phase_smear_s=0.0,
        plane_interval_s=0.0,
        typical_rate_hz=(10.0, 100.0),
        resolution_note="mesoscale (region-level), single plane",
        dynamics_note="single plane -> no slice smear; region pooling is the "
                      "dominant spatial approximation",
        verified=False,
    ),
    "bold_epi": ImagingMethod(
        name="bold_epi",
        label="BOLD fMRI (EPI / fast imaging sequence)",
        volume_integration_s=None,
        slice_phase_smear_s=None,
        plane_interval_s=None,
        typical_rate_hz=(0.2, 3.0),
        resolution_note="millimetre-scale voxels; the frame IS one TR, so the "
                        "sampling interval equals the integration window",
        dynamics_note=(
            "the observable is a vascular low-pass of neural activity: the "
            "effective HRF (peak ~5 s human, earlier in rodents) sets the "
            "resolvable timescale, and a per-corpus HRF must be fitted rather "
            "than imported"),
        source="Friston et al. 2003, NeuroImage 19:1273-1302 (forward model); "
               "TR values are corpus-specific",
        verified=True,
    ),
    "electrical": ImagingMethod(
        name="electrical",
        label="Electrical recording (EEG/ECoG/MEG/ephys)",
        volume_integration_s=None,
        slice_phase_smear_s=0.0,
        plane_interval_s=0.0,
        typical_rate_hz=(250.0, 30000.0),
        resolution_note="sensor-level or region-level; no optical reporter",
        dynamics_note="no indicator kinetics; the amplifier anti-alias filter "
                      "is the low-pass and is usually documented",
        verified=False,
    ),
}


@dataclass(frozen=True)
class Reporter:
    """What the fluorescence encodes, and how fast/linearly it does so."""

    name: str
    family: str
    readout: str
    localization: str
    tau_off_s: Optional[float]
    """Decay time constant of the indicator (s); ``None`` = not established."""
    hill_h: Optional[float]
    hill_kd: Optional[float]
    dynamics_valid: bool
    note: str
    source: Optional[str] = None
    verified: bool = False


def _reporter(name: str, **kw) -> Reporter:
    return Reporter(name=name, **kw)


REPORTERS: Dict[str, Reporter] = {
    # ---- calcium, cytoplasmic GCaMP family -------------------------------
    "gcamp6f": _reporter(
        "gcamp6f", family="gcamp", readout="dff", localization="cytoplasmic",
        tau_off_s=0.17, hill_h=None, hill_kd=None, dynamics_valid=True,
        note="fast GCaMP6 variant; half-decay ~150-180 ms in vivo",
        source="Chen et al. 2013; Nature 2023 (s41586-023-05828-9) comparison",
        verified=True),
    "gcamp6s": _reporter(
        "gcamp6s", family="gcamp", readout="dff", localization="cytoplasmic",
        tau_off_s=0.35, hill_h=None, hill_kd=None, dynamics_valid=True,
        note="slow/high-affinity variant; slower decay, better for small "
             "events, worse for spike-timing claims",
        source="Chen et al. 2013; Nature 2023 (s41586-023-05828-9) comparison",
        verified=False),
    "gcamp7f": _reporter(
        "gcamp7f", family="gcamp", readout="dff", localization="cytoplasmic",
        tau_off_s=0.18, hill_h=None, hill_kd=None, dynamics_valid=True,
        note="half-decay ~182 ms (Janelia characterisation)",
        source="Janelia jGCaMP8 characterisation table",
        verified=True),
    "jgcamp8f": _reporter(
        "jgcamp8f", family="gcamp", readout="dff", localization="cytoplasmic",
        tau_off_s=0.067, hill_h=None, hill_kd=None, dynamics_valid=True,
        note="fastest jGCaMP8 variant (half-decay ~67-84 ms in vivo)",
        source="Nature 2023 (s41586-023-05828-9); Janelia tables",
        verified=True),
    "jgcamp8s": _reporter(
        "jgcamp8s", family="gcamp", readout="dff", localization="cytoplasmic",
        tau_off_s=0.2, hill_h=None, hill_kd=None, dynamics_valid=True,
        note="sensitive jGCaMP8 variant (half-decay ~200 ms)",
        source="Nature 2023 (s41586-023-05828-9)",
        verified=True),
    # ---- calcium, nuclear-localised --------------------------------------
    "h2b_gcamp6s": _reporter(
        "h2b_gcamp6s", family="nuclear_gcamp", readout="dff",
        localization="nuclear", tau_off_s=None, hill_h=None, hill_kd=None,
        dynamics_valid=True,
        note="nuclear-targeted GCaMP6s: nucleocytoplasmic exchange adds a slow "
             "component on top of the indicator, so the effective low-pass is "
             "slower than the cytoplasmic parent - fit tau, do not import it",
        verified=False),
    "h2b_gcamp7f": _reporter(
        "h2b_gcamp7f", family="nuclear_gcamp", readout="dff",
        localization="nuclear", tau_off_s=None, hill_h=None, hill_kd=None,
        dynamics_valid=True,
        note="nuclear GCaMP7f (Tg(elavl3:H2B-GCaMP7f)): stable cell identity, "
             "and the nuclear kinetics are the slower component - the ZAPBench "
             "authors chose it precisely because a ~1 Hz volume rate cannot "
             "follow cytoplasmic GCaMP transients. Kinetics differ from the "
             "cytoplasmic parent, so fit tau instead of importing it",
        source="ZAPBench, arXiv:2503.02618 (Methods)",
        verified=True),
    # ---- calcium, FRET cameleon (the C. elegans salt corpus) --------------
    "yc2.60": _reporter(
        "yc2.60", family="fret_cameleon", readout="ratio",
        localization="nuclear", tau_off_s=None, hill_h=None, hill_kd=None,
        dynamics_valid=True,
        note="YFP/CFP ratio (excitation-ratiometric FRET); the released corpus "
             "is per-neuron z-scored after median filtering and detrending, so "
             "the absolute operating point is not recoverable - only "
             "scale-free comparisons are valid",
        source="Toyoshima et al. 2024, PLOS Comput Biol 1011848 (Methods)",
        verified=True),
    "rcamp": _reporter(
        "rcamp", family="red_gcamp", readout="dff", localization="cytoplasmic",
        tau_off_s=None, hill_h=None, hill_kd=None, dynamics_valid=True,
        note="red-shifted calcium indicator; useful for dual-colour "
             "co-expression, kinetics differ from green GCaMPs"),
    # ---- voltage (GEVI) ---------------------------------------------------
    "positron2_kv": _reporter(
        "positron2_kv", family="gevi", readout="voltage",
        localization="membrane", tau_off_s=0.00051, hill_h=None, hill_kd=None,
        dynamics_valid=True,
        note="chemigenetic voltage indicator, ~0.5-0.6 ms activation/"
             "deactivation; the readout is roughly linear in V, so no Hill "
             "saturation term applies (use a low-pass only)",
        source="Wang et al. 2023 (zebrafish whole-brain voltage imaging)",
        verified=True),
    "voltron": _reporter(
        "voltron", family="gevi", readout="voltage", localization="membrane",
        tau_off_s=0.00078, hill_h=None, hill_kd=None, dynamics_valid=True,
        note="Voltron525: tau_on ~64 us, tau_off ~0.78 ms, slower relaxation "
             "~4 ms; sub-millisecond, so the imaging rate decides what is "
             "recoverable, not the indicator",
        source="Abdelfattah et al. 2019 (Voltron)",
        verified=True),
    "arch": _reporter(
        "arch", family="gevi", readout="voltage", localization="membrane",
        tau_off_s=0.00017, hill_h=None, hill_kd=None, dynamics_valid=True,
        note="Arch/Archon: fast component sub-ms but a ~7 ms slow component; "
             "biphasic responses must not be modelled as a single low-pass",
        source="Genetic voltage indicators review (PMC6739974)",
        verified=True),
    # ---- vascular (BOLD fMRI) ---------------------------------------------
    "bold_hemodynamic": _reporter(
        "bold_hemodynamic", family="hemodynamic", readout="bold",
        localization="vascular", tau_off_s=None, hill_h=None, hill_kd=None,
        dynamics_valid=True,
        note="BOLD is a vascular response to a drive in arbitrary units: the "
             "link between the internal state and neural activity is learned, "
             "so the drive scale is a fitted gain and only the response shape "
             "(effective HRF peak/width/undershoot) is comparable across "
             "corpora. Pseudo-HRFs are initialised from Friston's Balloon-"
             "Windkessel reference",
        source="Friston, Harrison & Penny 2003, NeuroImage 19:1273-1302",
        verified=True),
    # ---- biochemical / static --------------------------------------------
    "perk": _reporter(
        "perk", family="erk", readout="static", localization="somatic",
        tau_off_s=600.0, hill_h=None, hill_kd=None, dynamics_valid=False,
        note="phospho-ERK immunohistochemistry in fixed tissue integrates "
             "activity over minutes (peak ~5-8 min, decay ~10-15 min) and "
             "reflects the history before fixation: it is a state marker, not "
             "a time series, so windowed dynamics objectives do not apply",
        source="PMC2614750; PMC1899229; PMC12095524",
        verified=True),
}


ELECTRICAL_REPORTER = Reporter(
    name="none", family="electrical", readout="absolute", localization="sensor",
    tau_off_s=None, hill_h=None, hill_kd=None, dynamics_valid=True,
    note="electrical recording: no fluorescent reporter (the amplifier "
         "anti-alias filter is the low-pass)")


@dataclass(frozen=True)
class SensorSpec:
    """A profile's binding of (modality, imaging method, reporter, readout)."""

    modality: str
    imaging: ImagingMethod
    reporter: Reporter
    readout: str
    calibration_required: bool
    notes: tuple = field(default_factory=tuple)
    hrf: Optional[Mapping[str, float]] = None
    """Haemodynamic initialisation for a BOLD readout (Friston's Table 1
    priors unless the profile overrides them); the emission then *learns* the
    effective response from there."""

    # ---- derived quantities used by the data/model contracts --------------
    @property
    def dynamics_valid(self) -> bool:
        return self.reporter.dynamics_valid and self.readout != "static"

    @property
    def emission_tau_s(self) -> Optional[float]:
        """Initial low-pass time constant for the observation channel."""
        if self.readout == "static":
            return None
        if self.reporter.tau_off_s is not None:
            return float(self.reporter.tau_off_s)
        return None

    @property
    def min_resolvable_lag_s(self) -> Optional[float]:
        """Best-case lag floor: two neurons in the same plane."""
        candidates = [c for c in (self.imaging.plane_interval_s,
                                  self.imaging.volume_integration_s) if c]
        return min(candidates) if candidates else None

    @property
    def cross_plane_lag_floor_s(self) -> Optional[float]:
        """Worst-case floor: phase offset between planes of one volume."""
        return self.imaging.slice_phase_smear_s

    def as_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "imaging": self.imaging.name,
            "reporter": self.reporter.name,
            "readout": self.readout,
            "localization": self.reporter.localization,
            "hrf": dict(self.hrf) if self.hrf else None,
            "emission_tau_s": self.emission_tau_s,
            "dynamics_valid": self.dynamics_valid,
            "min_resolvable_lag_s": self.min_resolvable_lag_s,
            "cross_plane_lag_floor_s": self.cross_plane_lag_floor_s,
            "calibration_required": self.calibration_required,
            "notes": list(self.notes),
        }


def resolve_sensor(modality: str, spec: Optional[Mapping[str, Any]]) -> SensorSpec:
    """Validate a profile's sensor block for one modality.

    Raises on unknown names and on physically inconsistent combinations
    (a voltage reporter declared as calcium, a Hill readout without a reporter
    that saturates, a static marker used for dynamics).
    """
    spec = dict(spec or {})
    imaging_name = spec.get("imaging")
    reporter_name = spec.get("reporter")
    if imaging_name is None and reporter_name is None:
        raise ValueError(
            f"modality {modality!r}: sensor block needs at least 'imaging' or "
            f"'reporter' (known imaging: {sorted(IMAGING_METHODS)})")
    if imaging_name is None:
        imaging_name = ("electrical" if modality in
                        ("eeg", "ecog", "meg", "fmri", "behavior") else None)
        if imaging_name is None:
            raise ValueError(
                f"modality {modality!r}: 'imaging' is required for optical "
                f"modalities; known: {sorted(IMAGING_METHODS)}")
    if imaging_name not in IMAGING_METHODS:
        raise ValueError(
            f"unknown imaging method {imaging_name!r}; known: "
            f"{sorted(IMAGING_METHODS)}")
    imaging = IMAGING_METHODS[imaging_name]

    if reporter_name is None:
        if imaging.name != "electrical":
            raise ValueError(
                f"modality {modality!r}: 'reporter' is required for optical "
                f"imaging {imaging_name!r}; known: {sorted(REPORTERS)}")
        return SensorSpec(
            modality=modality, imaging=imaging, reporter=ELECTRICAL_REPORTER,
            readout=spec.get("readout", "absolute"),
            calibration_required=False,
            notes=("electrical modality: reporter parameters do not apply",))
    if reporter_name not in REPORTERS:
        raise ValueError(
            f"unknown reporter {reporter_name!r}; known: {sorted(REPORTERS)}")
    reporter = REPORTERS[reporter_name]
    readout = str(spec.get("readout", reporter.readout))
    if readout not in READOUT_KINDS:
        raise ValueError(
            f"readout {readout!r} must be one of {READOUT_KINDS}")
    if readout == "bold" and reporter.family != "hemodynamic":
        raise ValueError(
            f"modality {modality!r}: readout 'bold' requires the haemodynamic "
            f"reporter, got {reporter.name!r} ({reporter.family})")
    if readout == "voltage" and reporter.family != "gevi":
        raise ValueError(
            f"modality {modality!r}: readout 'voltage' requires a voltage "
            f"reporter, got {reporter.name!r} ({reporter.family})")
    if imaging.name == "bold_epi" and reporter.family != "hemodynamic":
        raise ValueError(
            f"imaging 'bold_epi' measures a vascular signal: the reporter must "
            f"be haemodynamic, got {reporter.name!r} ({reporter.family})")
    if modality == "fmri" and reporter.family != "hemodynamic":
        raise ValueError(
            f"modality 'fmri' needs a haemodynamic (BOLD) reporter, got "
            f"{reporter.name!r}; a calcium/voltage reporter under fmri would "
            f"silently reinterpret the observable")
    if readout == "bold" and modality != "fmri":
        raise ValueError(
            f"readout 'bold' belongs to the fmri modality; got {modality!r}")
    if modality == "calcium" and reporter.family == "gevi":
        raise ValueError(
            f"modality 'calcium' cannot use the voltage reporter "
            f"{reporter.name!r}; use the voltage modality")
    if readout == "static" and reporter.dynamics_valid:
        raise ValueError(
            f"readout 'static' requires a fixed-tissue/history marker "
            f"(e.g. 'perk'); {reporter.name!r} reports a dynamic signal")
    if readout != "static" and not reporter.dynamics_valid:
        raise ValueError(
            f"reporter {reporter.name!r} is a fixed-tissue history marker and "
            f"cannot back a dynamic readout ({readout!r}); declare "
            f"readout='static' and exclude it from the dynamics contract")
    notes = []
    if reporter.localization == "nuclear" and reporter.tau_off_s is None:
        notes.append("nuclear localisation adds an unmeasured slow component: "
                     "fit the emission tau instead of importing it")
    if reporter.tau_off_s is None and readout != "static":
        notes.append("reporter kinetics not established in the registry: "
                     "calibration required")
    spec_tau = spec.get("tau_s")
    if spec_tau is not None:
        notes.append(f"profile overrides emission tau with {spec_tau} s")
    calibration_required = (
        reporter.tau_off_s is None and readout != "static")
    hrf = spec.get("hrf")
    if readout == "bold" and not hrf:
        notes.append("no 'hrf' block given: initialised from Friston's Table 1 "
                     "priors and fitted from there")
    return SensorSpec(
        modality=modality, imaging=imaging, reporter=reporter, readout=readout,
        calibration_required=calibration_required, notes=tuple(notes),
        hrf=dict(hrf) if hrf else None)


def sensor_spec_from_profile(
    profile_sensors: Optional[Mapping[str, Mapping[str, Any]]],
    modalities,
) -> Dict[str, SensorSpec]:
    """Resolve every declared sensor block; undeclared modalities stay absent."""
    out: Dict[str, SensorSpec] = {}
    for modality, block in (profile_sensors or {}).items():
        if modality not in tuple(modalities):
            raise ValueError(
                f"sensor block declared for {modality!r}, which is not one of "
                f"the profile modalities {tuple(modalities)}")
        out[modality] = resolve_sensor(modality, block)
    return out
