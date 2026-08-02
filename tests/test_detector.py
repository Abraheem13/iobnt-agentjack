"""Gate 2 as unit tests: modulation, framing and detection."""

from __future__ import annotations

import numpy as np
import pytest

from agentjack.channel.diffusion import ChannelParams, DiffusionChannel
from agentjack.channel.isi import apply_isi, build_isi_matrix, isi_severity, subtract_isi
from agentjack.physical.detector import (
    DecisionFeedbackDetector,
    GRUDetector,
    ThresholdDetector,
    ber,
)
from agentjack.physical.modulation import (
    PREAMBLE,
    SYMBOL_BUDGET,
    CSK,
    OOK,
    CommandCodebook,
    bits_to_int,
    int_to_bits,
)

# ---------------------------------------------------------------- modulation


def test_bit_int_roundtrip():
    for v in [0, 1, 7, 255, 1023]:
        assert bits_to_int(int_to_bits(v, 10)) == v
    with pytest.raises(ValueError):
        int_to_bits(1024, 10)


def test_ook_roundtrip():
    bits = np.array([1, 0, 1, 1, 0])
    assert np.array_equal(OOK().demodulate(OOK().modulate(bits)), bits)


def test_ook_rejects_non_binary():
    with pytest.raises(ValueError):
        OOK().modulate(np.array([0, 2, 1]))


def test_csk_roundtrip_and_levels():
    csk = CSK(levels=4)
    bits = np.array([1, 0, 1, 1, 0, 0])
    assert np.array_equal(csk.demodulate(csk.modulate(bits)), bits)
    with pytest.raises(ValueError):
        CSK(levels=3)


# ----------------------------------------------------------------- codebook


def test_frame_fits_the_symbol_budget():
    cb = CommandCodebook()
    assert len(cb.encode("command", "high_dose", 42, 7)) == SYMBOL_BUDGET
    assert len(cb.frame("command", "high_dose")) == len(PREAMBLE) + SYMBOL_BUDGET


def test_codebook_roundtrip():
    cb = CommandCodebook()
    d = cb.decode(cb.encode("command", "high_dose", value=42, nonce=7))
    assert d["type"] == "command" and d["opcode"] == "high_dose"
    assert d["value"] == 42 and d["nonce"] == 7 and d["checksum_ok"]


def test_checksum_covers_the_whole_body():
    """Regression: an earlier checksum ignored the opcode field entirely, so
    flipping one bit silently turned high_dose into med_dose and still verified.
    """
    cb = CommandCodebook()
    payload = cb.encode("command", "high_dose", value=42, nonce=7)
    body_len = cb.N_TYPE + cb.N_OP + cb.N_VAL + cb.N_NONCE
    for i in range(body_len):
        t = payload.copy()
        t[i] ^= 1
        assert not cb.decode(t)["checksum_ok"], f"bit {i} flip went undetected"


def test_checksum_is_forgeable_by_design():
    """Unkeyed integrity is not authentication - the threat model depends on this."""
    cb = CommandCodebook()
    forged = cb.encode("command", "high_dose", value=999, nonce=3)
    assert cb.decode(forged)["checksum_ok"]


def test_codebook_rejects_unknown_fields():
    cb = CommandCodebook()
    with pytest.raises(ValueError):
        cb.encode("nonsense", "high_dose")
    with pytest.raises(ValueError):
        cb.encode("command", "self_destruct")


def test_budget_too_small_is_rejected():
    with pytest.raises(ValueError):
        CommandCodebook(budget=8)


# ---------------------------------------------------------------------- ISI


def test_isi_matrix_matches_convolution():
    cir = np.array([0.5, 0.3, 0.1])
    s = np.array([1.0, 0.0, 1.0, 1.0])
    assert np.allclose(build_isi_matrix(cir, len(s)) @ s, apply_isi(s, cir))


def test_isi_matrix_is_causal():
    H = build_isi_matrix(np.array([0.5, 0.3]), 4)
    assert np.allclose(H, np.tril(H))


def test_isi_severity_is_zero_without_tail():
    assert isi_severity(np.array([1.0, 0.0, 0.0])) == 0.0


def test_subtract_isi_removes_known_interference():
    cir = np.array([0.5, 0.3, 0.1])
    clean = subtract_isi(0.5 * 100 + 0.3 * 100, np.array([1.0, 0.0]), cir, 100.0)
    assert clean == pytest.approx(50.0)


# ---------------------------------------------------------------- detection


def _trace(N, n=6000, seed=0):
    rng = np.random.default_rng(seed)
    p = ChannelParams(n_molecules=N)
    ch = DiffusionChannel(p, seed=seed)
    bits = rng.integers(0, 2, n)
    return bits, ch.transmit(bits), ch


def test_ber_rejects_length_mismatch():
    with pytest.raises(ValueError):
        ber(np.array([1, 0]), np.array([1]))


def test_threshold_requires_fit():
    with pytest.raises(RuntimeError):
        ThresholdDetector().detect(np.array([1.0]))


def test_dfe_beats_threshold_on_an_isi_limited_channel():
    bits, obs, ch = _trace(8000)
    th = ThresholdDetector().fit(obs, bits)
    dfe = DecisionFeedbackDetector(ch.impulse_response(), 8000)
    assert ber(bits, dfe.detect(obs)) < ber(bits, th.detect(obs)) / 10


def test_dfe_ber_falls_with_molecule_budget():
    errs = []
    for N in [2000, 8000, 32000]:
        bits, obs, ch = _trace(N)
        errs.append(ber(bits, DecisionFeedbackDetector(ch.impulse_response(), N).detect(obs)))
    assert errs[0] > errs[1] >= errs[2]


def test_default_operating_point_supports_frame_delivery():
    """Regression: the original default gave 1.6% frame success, which would
    have silently broken the whole command pipeline on Day 4."""
    p = ChannelParams()
    bits, obs, ch = _trace(p.n_molecules, n=20000)
    b = ber(bits, DecisionFeedbackDetector(ch.impulse_response(), p.n_molecules).detect(obs))
    assert (1 - b) ** (len(PREAMBLE) + SYMBOL_BUDGET) >= 0.95


def test_gru_requires_fit():
    with pytest.raises(RuntimeError):
        GRUDetector().detect(np.array([1.0]))


@pytest.mark.slow
def test_gru_learns_the_channel():
    bits, obs, _ = _trace(16000, n=20000)
    gru = GRUDetector(window=16, hidden_size=48).fit(obs, bits, epochs=15)
    assert ber(bits, gru.detect(obs)) < 0.1
