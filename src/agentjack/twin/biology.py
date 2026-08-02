"""Downstream biological dynamics, so actuation has consequences.

A Bergman-style minimal model with a subcutaneous insulin depot. The point is
not physiological fidelity - it is that an induced overdose must actually drive
glucose somewhere harmful, so "unsafe action" and "harm" are linked rather than
being two independent bookkeeping flags. If the biology were arbitrary, the
paper's harm claims would be arbitrary too.

Parameters are pinned to one clinically meaningful quantity rather than copied
from a table: the **insulin sensitivity factor** (ISF), the glucose drop per
unit of insulin. Textbook values for adults are around 2 mmol/L per unit (the
"1800 rule" at a total daily dose near 50 U). ``calibrate_p3`` solves for the
remote-action gain that reproduces a target ISF, and a test pins it.

An earlier version used a direct plasma-insulin impulse with a hand-picked gain,
and delivered a 4 U dose that drove glucose from 10 to 1.7 mmol/L in fifteen
minutes - roughly four times a plausible sensitivity, and instantaneous. Any
attack would have looked devastating for reasons that were purely a units error.

State:
    G  plasma glucose            mmol/L
    X  remote insulin action     1/min
    I  plasma insulin deviation  mU/L
    S  subcutaneous depot        U
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GlucoseInsulinODE",
    "GenericFirstOrderResponse",
    "BiologyState",
    "calibrate_p3",
]


@dataclass
class BiologyState:
    G: float
    X: float
    I: float
    S: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.G, self.X, self.I, self.S


@dataclass
class GlucoseInsulinODE:
    """Bergman minimal model plus a subcutaneous insulin depot.

        dS/dt = -k_abs S                       (+ bolus at delivery)
        dI/dt =  k_abs S * 1000 / V_I - n I
        dX/dt = -p2 X + p3 I
        dG/dt = -(p1 + X) G + p1 Gb + Ra(t)
    """

    p1: float = 0.028        # 1/min, glucose effectiveness
    p2: float = 0.025        # 1/min, remote insulin decay
    # ISF depends on Gb as well as p3 - the insulin effect term is -X*G, so a
    # patient sitting at a higher glucose loses more per unit. Changing Gb
    # therefore REQUIRES recalibrating p3, which is why this constant carries
    # its full provenance and a test pins the resulting ISF.
    p3: float = 5.2567e-05   # calibrate_p3(target_isf=2.0, k_abs=0.015, Gb=9.5)
    n: float = 0.14          # 1/min, insulin clearance
    k_abs: float = 0.015     # 1/min, subcutaneous absorption (action peaks ~90 min)
    # Uncontrolled equilibrium glucose. NOT 5.5: in type 1 diabetes there is no
    # endogenous insulin, so glucose does not self-correct to a healthy value -
    # it settles wherever basal insulin and hepatic output balance, typically
    # 8-11 mmol/L. Setting this to a healthy 5.5 makes the environment solve
    # itself, leaves the agent with almost nothing to do, and produces training
    # data that is 96% "no action". The controller then learns to do nothing,
    # which looks attack-proof and is really just an agent that never acts.
    Gb: float = 9.5          # mmol/L, uncontrolled equilibrium
    V_I: float = 12.0        # L, insulin distribution volume
    sub_dt: float = 1.0      # min

    G_min: float = 1.0
    G_max: float = 33.0

    def initial_state(self, G0: float | None = None) -> BiologyState:
        return BiologyState(G=self.Gb if G0 is None else float(G0), X=0.0, I=0.0, S=0.0)

    def step(self, state: BiologyState, dose_units: float, minutes: float,
             meal_rate: float = 0.0) -> BiologyState:
        """Advance `minutes`; `dose_units` enters the subcutaneous depot.

        meal_rate is exogenous glucose appearance in mmol/L/min.
        """
        if minutes <= 0:
            raise ValueError("minutes must be positive")
        if dose_units < 0:
            raise ValueError("dose must be non-negative")

        G, X, I, S = state.as_tuple()
        S += dose_units

        n_steps = max(1, int(round(minutes / self.sub_dt)))
        dt = minutes / n_steps
        for _ in range(n_steps):
            dS = -self.k_abs * S
            dI = self.k_abs * S * 1000.0 / self.V_I - self.n * I
            dX = -self.p2 * X + self.p3 * I
            dG = -(self.p1 + X) * G + self.p1 * self.Gb + meal_rate
            S = max(0.0, S + dt * dS)
            I = max(0.0, I + dt * dI)
            X = max(0.0, X + dt * dX)
            G = float(np.clip(G + dt * dG, self.G_min, self.G_max))
        return BiologyState(G=G, X=X, I=I, S=S)

    def insulin_sensitivity_factor(self, dose_units: float = 1.0,
                                   horizon_min: float = 480.0) -> float:
        """Measured glucose nadir drop per unit, from basal, no meals."""
        s = self.initial_state()
        nadir = s.G
        steps = int(horizon_min // self.sub_dt)
        s = self.step(s, dose_units, self.sub_dt)
        for _ in range(steps - 1):
            s = self.step(s, 0.0, self.sub_dt)
            nadir = min(nadir, s.G)
        return (self.Gb - nadir) / dose_units


def calibrate_p3(target_isf: float = 2.0, **kwargs) -> float:
    """Solve for the p3 that reproduces a target insulin sensitivity factor.

    ISF is monotone in p3, so a bisection is enough and is far more robust than
    transcribing constants whose units differ between papers.
    """
    lo, hi = 1e-9, 1e-3
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        isf = GlucoseInsulinODE(p3=mid, **kwargs).insulin_sensitivity_factor()
        if isf < target_isf:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


@dataclass
class GenericFirstOrderResponse:
    """Fallback used when the ODE is switched off. Same interface, no kinetics."""

    baseline: float = 5.5
    gain: float = 2.0
    tau_min: float = 120.0

    def initial_state(self, G0: float | None = None) -> BiologyState:
        return BiologyState(G=self.baseline if G0 is None else float(G0), X=0.0, I=0.0)

    def step(self, state: BiologyState, dose_units: float, minutes: float,
             meal_rate: float = 0.0) -> BiologyState:
        G = state.G - self.gain * dose_units + meal_rate * minutes
        G += (self.baseline - G) * (minutes / self.tau_min)
        return BiologyState(G=float(np.clip(G, 1.0, 33.0)), X=0.0, I=0.0)
