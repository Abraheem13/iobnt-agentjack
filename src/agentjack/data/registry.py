"""Single source of truth for every dataset used in the paper.

Nothing downloads or loads data except through this registry, so the paper's
data-availability statement can be generated mechanically from it.

Tiers
-----
A  Real MC testbed traces      -> calibrate + validate the digital twin
B  Biosensor telemetry         -> drive the agent's actuation decisions
C  Prompt-injection corpora    -> semantic payload material for attack A4
D  Systems-biology models      -> downstream actuation dynamics (optional)
E  Synthetic                   -> primary training/eval corpus from our twin
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Dataset:
    key: str
    tier: str
    title: str
    source: str
    citation: str
    access: str            # "open" | "registration" | "gated"
    role: str              # what it does for THIS paper
    loader: str
    notes: str = ""
    tags: list[str] = field(default_factory=list)


REGISTRY: dict[str, Dataset] = {
    # ------------------------------------------------------------------ Tier A
    "macroscale_ethanol": Dataset(
        key="macroscale_ethanol",
        tier="A",
        title="Dataset for Macroscale Molecular Communication Testbed",
        source="https://ieee-dataport.org/documents/dataset-macroscale-molecular-communication-testbed",
        citation="Hofmann, Torres Gomez, Fitzek, Dressler, IEEE DataPort, 2023, doi:10.21227/ytkm-xp81",
        access="registration",
        role="PRIMARY channel calibration. Ethanol released by propulsive mechanism, "
             "drift-assisted free-space channel, COTS alcohol sensor sampled every 0.1 s. "
             "Fits the CIR shape, ISI tail and sensor noise of the digital twin.",
        loader="agentjack.data.loaders.mc_testbed:load_macroscale_ethanol",
        notes="Ships MATLAB processing/visualisation code; we re-implement in Python.",
        tags=["cir", "isi", "sensor-noise", "drift"],
    ),
    "analog_network_coding": Dataset(
        key="analog_network_coding",
        tier="A",
        title="Dataset for Analog Network Coding in Molecular Communications",
        source="https://ieee-dataport.org/documents/dataset-analog-network-coding-molecular-communications-practical-implementation",
        citation="IEEE DataPort (macroscale MC testbed for analog network coding)",
        access="registration",
        role="SPOOFING realism. Multiple transmitters superposing in one channel is "
             "exactly the physical situation attack A2 assumes. Propagation_delay.csv "
             "grounds the attacker/legitimate timing offsets.",
        loader="agentjack.data.loaders.mc_testbed:load_analog_network_coding",
        tags=["multi-tx", "superposition", "propagation-delay"],
    ),
    "proton_pump_bacteria": Dataset(
        key="proton_pump_bacteria",
        tier="A",
        title="A Molecular Communication Testbed Based on Proton Pumping Bacteria",
        source="https://ieee-dataport.org/open-access/molecular-communication-testbed-based-proton-pumping-bacteria",
        citation="Grebenstein, Kirchner, Wicke, Ahmadzadeh, Jamali, Fischer, Weigel, "
                 "Burkovski, Schober, IEEE DataPort, 2019, doi:10.21227/3zj6-pm05 "
                 "(methods and data published in IEEE TMBMC)",
        access="registration",
        role="BIO-CYBER INTERFACE realism. A biological optical-to-chemical transducer "
             "with 18 signal experiments and 12 dedicated noise measurements. Gives the "
             "gateway a measured biological response and a measured noise floor. "
             "Strategically valuable: the dataset paper appeared in TMBMC itself.",
        loader="agentjack.data.loaders.mc_testbed:load_proton_pump_bacteria",
        notes="Open-access listing, but IEEE DataPort access is required; free for IEEE members.",
        tags=["bio-transducer", "noise-floor", "tmbmc"],
    ),
    "magnetic_nanoparticle_flow": Dataset(
        key="magnetic_nanoparticle_flow",
        tier="A",
        title="Channel Parameter Studies with a Biocompatible Testbed (magnetic nanoparticles, duct flow)",
        source="https://www.techrxiv.org/doi/full/10.36227/techrxiv.22674685.v1",
        citation="Bartunik, Unterweger, Kirchner, Fischer et al. (biocompatible MC testbed, "
                 "magnetic nanoparticles in duct flow); data provided as a public supplement",
        access="open",
        role="FLOW channel + LONG SEQUENCES. A wide parameter sweep plus a large binary "
             "transmission sequence - the only Tier-A source long enough to train and "
             "evaluate the learned detector on real traces.",
        loader="agentjack.data.loaders.mc_testbed:load_magnetic_nanoparticle",
        tags=["flow", "long-sequence", "parameter-sweep", "biocompatible"],
    ),
    # ------------------------------------------------------------------ Tier B
    "shanghai_cgm": Dataset(
        key="shanghai_cgm",
        tier="B",
        title="Diabetes Datasets: ShanghaiT1DM and ShanghaiT2DM",
        source="https://doi.org/10.6084/m9.figshare.c.6310860",
        citation="Zhao, Zhu, Wang, Rao. Chinese diabetes datasets for data-driven machine "
                 "learning. Scientific Data, 2023.",
        access="open",
        role="PRIMARY task telemetry. 12 T1DM + 100 T2DM patients, 3-14 days of CGM at "
             "15-min resolution with dietary records, clinical characteristics and "
             "medications. Makes the glycemic-control actuation loop clinically real and "
             "gives the safety envelope meaningful physiological consequences.",
        loader="agentjack.data.loaders.cgm:load_shanghai",
        notes="Figshare, no DUA - downloadable on Day 1. Split by patient, never within.",
        tags=["cgm", "time-series", "clinical", "open"],
    ),
    "physionet_sepsis_2019": Dataset(
        key="physionet_sepsis_2019",
        tier="B",
        title="PhysioNet/CinC Challenge 2019: Early Prediction of Sepsis",
        source="https://physionet.org/content/challenge-2019/",
        citation="Reyna et al., PhysioNet/Computing in Cardiology Challenge 2019",
        access="open",
        role="SECOND TASK. Hourly vitals and labs across ~40k ICU stays. Shows the "
             "molecular-prompt-injection threat is not specific to one clinical loop; "
             "the attacker goal here is alarm suppression rather than overdose.",
        loader="agentjack.data.loaders.physionet:load_sepsis_challenge",
        tags=["vitals", "time-series", "generalisation", "open"],
    ),
    # ------------------------------------------------------------------ Tier C
    "agentdojo": Dataset(
        key="agentdojo",
        tier="C",
        title="AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses",
        source="https://github.com/ethz-spylab/agentdojo",
        citation="Debenedetti et al., NeurIPS 2024. 97 realistic tasks, 629 security test cases.",
        access="open",
        role="Payload templates AND methodology. We adopt its joint utility/security "
             "reporting convention so our numbers are legible to the agent-security "
             "community - a deliberate cross-community citation bridge.",
        loader="agentjack.data.loaders.injection_corpora:load_agentdojo",
        notes="We reuse payload text and evaluation convention, not its environments.",
        tags=["prompt-injection", "agent-security", "methodology"],
    ),
    "injecagent": Dataset(
        key="injecagent",
        tier="C",
        title="InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated LLM Agents",
        source="https://github.com/uiuc-kang-lab/InjecAgent",
        citation="Zhan et al., 2024. 1,054 test cases, 17 user tools, 62 attacker tools.",
        access="open",
        role="Attacker-goal taxonomy (Direct Harm / Data Stealing) mapped onto molecular "
             "actuation goals. Its 'enhanced attack' prefixes are the strongest short "
             "payloads, which matters under our symbol budget.",
        loader="agentjack.data.loaders.injection_corpora:load_injecagent",
        tags=["prompt-injection", "taxonomy"],
    ),
    "bipia": Dataset(
        key="bipia",
        tier="C",
        title="BIPIA: Benchmark for Indirect Prompt Injection Attacks",
        source="https://github.com/microsoft/BIPIA",
        citation="Yi et al. 250 attacker objectives across five scenarios, plus defenses.",
        access="open",
        role="Defense baselines. Supplies the black-box/white-box mitigations that our "
             "D2 guardrail baseline implements, so the comparison is against published "
             "defenses rather than a strawman.",
        loader="agentjack.data.loaders.injection_corpora:load_bipia",
        tags=["prompt-injection", "defense-baselines"],
    ),
    "agent_security_bench": Dataset(
        key="agent_security_bench",
        tier="C",
        title="Agent Security Bench (ASB)",
        source="https://github.com/agiresearch/ASB",
        citation="Zhang et al., 2025. 16 attack types against 11 defenses across 10 scenarios.",
        access="open",
        role="Coverage check: confirms our four molecular attack classes map onto a "
             "recognised attack taxonomy, and supplies additional defense baselines.",
        loader="agentjack.data.loaders.injection_corpora:load_asb",
        tags=["prompt-injection", "coverage"],
    ),
    # ------------------------------------------------------------------ Tier D
    "biomodels_glucose_insulin": Dataset(
        key="biomodels_glucose_insulin",
        tier="D",
        title="BioModels curated glucose-insulin ODE models",
        source="https://www.ebi.ac.uk/biomodels/",
        citation="BioModels Database (curated SBML)",
        access="open",
        role="OPTIONAL actuation dynamics. Gives 'release drug' a mechanistic downstream "
             "effect instead of a lookup table. Behind a feature flag - this is the first "
             "thing to cut if the schedule slips.",
        loader="agentjack.data.loaders.biomodels:load_sbml",
        notes="CUT LINE candidate. Fallback: twin.biology.GenericFirstOrderResponse.",
        tags=["sbml", "ode", "optional"],
    ),
    # ------------------------------------------------------------------ Tier E
    "synthetic_twin_corpus": Dataset(
        key="synthetic_twin_corpus",
        tier="E",
        title="AgentJack synthetic MC corpus (generated by this repo)",
        source="generated: python scripts/run_experiment.py --generate-corpus",
        citation="This work.",
        access="open",
        role="PRIMARY training and evaluation corpus. Calibrated against Tier A, so it "
             "inherits real channel statistics while giving us unlimited labelled "
             "attack/benign episodes and full ground truth for detection metrics.",
        loader="agentjack.twin.digital_twin:BioCyberTwin",
        notes="Released with fixed seeds so every number in the paper is reproducible.",
        tags=["synthetic", "primary"],
    ),
}


def by_tier(tier: str) -> list[Dataset]:
    return [d for d in REGISTRY.values() if d.tier == tier]


def data_availability_statement() -> str:
    """Generate the paper's data-availability paragraph from the registry."""
    raise NotImplementedError  # Day 14
