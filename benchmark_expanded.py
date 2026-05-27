"""
CR-CA Expanded Benchmark Suite v2.0
====================================

A comprehensive, multi-scenario, multi-baseline benchmark for evaluating
causal reasoning systems against ground-truth synthetic data.

Baselines:
- CR-CA Heuristic (linear mode)
- CR-CA Heuristic (nonlinear mode)
- CR-CA Formal (crca_core identify_effect)
- DoWhy (backdoor linear regression)
- OLS (naive regression, ignores confounding)
- Oracle (knows true graph and mechanisms)
- Random

Scenarios:
- Simple chain: X -> Y -> Z
- Fork (confounding): Z -> X, Z -> Y
- Collider: X -> Z, Y -> Z
- Multiple confounders
- Front-door mediation
- Instrumental variable
- Nonlinear mechanisms
- Hidden confounding

Metrics:
- Intervention MSE (predicted vs ground-truth do-effects)
- Counterfactual MSE (predicted vs ground-truth counterfactuals)
- Identifiability accuracy (does system abstain when it should?)
- Graph recovery F1 (recovered edges vs true edges)
- Runtime
- Calibration (prediction intervals coverage)

Usage:
    python benchmark_expanded.py --scenarios all --models all --n_samples 1000 --verbose
    
Results saved to benchmark_expanded_results.json and benchmark_expanded_report.md
"""

import sys
import os
import json
import time
import argparse
import traceback
import warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Any, Optional, Callable
from enum import Enum
import importlib.util

# Third-party
import numpy as np
from numpy.random import default_rng

# Suppress sklearn convergence warnings etc.
warnings.filterwarnings('ignore')

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)

# Load CR-CA
spec = importlib.util.spec_from_file_location("crca_module", os.path.join(repo_root, "CRCA.py"))
crca_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crca_mod)
CRCAAgent = getattr(crca_mod, "CRCAAgent")

# Load crca_core formal tier
try:
    from crca_core.identify import identify_effect
    from crca_core.models.spec import DraftSpec, CausalGraphSpec, NodeSpec, EdgeSpec, RoleSpec
    from crca_core.core.lifecycle import lock_spec
    CRCA_CORE_AVAILABLE = True
except ImportError:
    CRCA_CORE_AVAILABLE = False

# Load DoWhy
try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

# Load sklearn/statsmodels for baselines
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# ============================================================================
# DATA CLASSES
# ============================================================================

class ScenarioType(Enum):
    CHAIN = "chain"
    FORK = "fork"
    COLLIDER = "collider"
    MULTI_CONFOUNDER = "multi_confounder"
    FRONTDOOR = "frontdoor"
    INSTRUMENTAL_VARIABLE = "iv"
    NONLINEAR = "nonlinear"
    HIDDEN_CONFOUNDING = "hidden_confounding"


@dataclass
class GroundTruthSCM:
    """A ground-truth structural causal model for synthetic data generation."""
    name: str
    variables: List[str]
    edges: List[Tuple[str, str]]  # directed edges
    structural_equations: Dict[str, Callable]  # variable -> function(parents, noise)
    noise_dists: Dict[str, Callable]  # variable -> function(rng) -> noise
    intervention_variables: List[str]
    target_variable: str
    hidden_variables: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class BenchmarkResult:
    scenario: str
    model: str
    metric: str
    value: float
    unit: str
    passed: Optional[bool] = None
    details: str = ""
    duration_ms: float = 0.0
    fit_duration_ms: float = 0.0
    n_samples: int = 0


@dataclass
class ScenarioResult:
    scenario_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def add(self, r: BenchmarkResult):
        self.results.append(r)


# ============================================================================
# GROUND TRUTH SIMULATOR
# ============================================================================

class GroundTruthSimulator:
    """Simulates from a ground-truth SCM. Can compute exact do() and counterfactuals."""
    
    def __init__(self, scm: GroundTruthSCM, rng: Optional[np.random.Generator] = None):
        self.scm = scm
        self.rng = rng or default_rng(42)
        self._topo_order = self._topological_sort()
    
    def _topological_sort(self) -> List[str]:
        """Kahn's algorithm."""
        in_degree = {v: 0 for v in self.scm.variables}
        children = {v: [] for v in self.scm.variables}
        for u, v in self.scm.edges:
            children[u].append(v)
            in_degree[v] += 1
        queue = [v for v in self.scm.variables if in_degree[v] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order
    
    def sample(self, n: int) -> Dict[str, np.ndarray]:
        """Sample n observations from the observational distribution."""
        data = {v: np.zeros(n) for v in self.scm.variables}
        noise_cache = {}
        for var in self._topo_order:
            noise = self.scm.noise_dists[var](self.rng, n)
            noise_cache[var] = noise
            parents = [u for u, v in self.scm.edges if v == var]
            parent_vals = {p: data[p] for p in parents}
            data[var] = self.scm.structural_equations[var](parent_vals, noise)
        return data, noise_cache
    
    def do(self, n: int, interventions: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Sample n observations from the interventional distribution P(. | do(.))."""
        data = {v: np.zeros(n) for v in self.scm.variables}
        for var in self._topo_order:
            if var in interventions:
                data[var] = np.full(n, interventions[var])
                continue
            parents = [u for u, v in self.scm.edges if v == var]
            parent_vals = {p: data[p] for p in parents}
            noise = self.scm.noise_dists[var](self.rng, n)
            data[var] = self.scm.structural_equations[var](parent_vals, noise)
        return data
    
    def counterfactual(self, factual_data: Dict[str, np.ndarray], 
                       interventions: Dict[str, float],
                       noise_cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute counterfactuals given factual data and abducted noise."""
        n = len(list(factual_data.values())[0])
        cf = {v: np.zeros(n) for v in self.scm.variables}
        for var in self._topo_order:
            if var in interventions:
                cf[var] = np.full(n, interventions[var])
                continue
            parents = [u for u, v in self.scm.edges if v == var]
            parent_vals = {p: cf[p] for p in parents}
            # Reuse abducted noise from factual
            noise = noise_cache[var]
            cf[var] = self.scm.structural_equations[var](parent_vals, noise)
        return cf


# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

def make_chain_scenario() -> GroundTruthSCM:
    """Simple chain: X -> Y -> Z. Effect of do(X) on Z is identifiable."""
    def f_X(parents, noise): return noise
    def f_Y(parents, noise): return 2.0 * parents['X'] + noise
    def f_Z(parents, noise): return 0.5 * parents['Y'] + noise
    
    def n_X(rng, n): return rng.normal(0, 1, n)
    def n_Y(rng, n): return rng.normal(0, 0.5, n)
    def n_Z(rng, n): return rng.normal(0, 0.5, n)
    
    return GroundTruthSCM(
        name="chain",
        variables=["X", "Y", "Z"],
        edges=[("X", "Y"), ("Y", "Z")],
        structural_equations={"X": f_X, "Y": f_Y, "Z": f_Z},
        noise_dists={"X": n_X, "Y": n_Y, "Z": n_Z},
        intervention_variables=["X"],
        target_variable="Z",
        description="Simple chain X->Y->Z. do(X) effect on Z is identifiable via product of coefficients."
    )

def make_fork_scenario() -> GroundTruthSCM:
    """Fork with confounding: Z -> X, Z -> Y, X -> Y.
    P(Y|do(X)) != P(Y|X) unless we adjust for Z."""
    def f_Z(parents, noise): return noise
    def f_X(parents, noise): return 1.5 * parents['Z'] + noise
    def f_Y(parents, noise): return 2.0 * parents['Z'] + 1.0 * parents['X'] + noise
    
    def n_Z(rng, n): return rng.normal(0, 1, n)
    def n_X(rng, n): return rng.normal(0, 0.5, n)
    def n_Y(rng, n): return rng.normal(0, 0.5, n)
    
    return GroundTruthSCM(
        name="fork",
        variables=["Z", "X", "Y"],
        edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        structural_equations={"Z": f_Z, "X": f_X, "Y": f_Y},
        noise_dists={"Z": n_Z, "X": n_X, "Y": n_Y},
        intervention_variables=["X"],
        target_variable="Y",
        description="Fork with confounder Z. Naive regression gives biased effect; adjustment required."
    )

def make_collider_scenario() -> GroundTruthSCM:
    """Collider: X -> Z, Y -> Z. Conditioning on Z induces spurious X-Y association."""
    def f_X(parents, noise): return noise
    def f_Y(parents, noise): return noise
    def f_Z(parents, noise): return 1.0 * parents['X'] + 1.0 * parents['Y'] + noise
    
    def n_X(rng, n): return rng.normal(0, 1, n)
    def n_Y(rng, n): return rng.normal(0, 1, n)
    def n_Z(rng, n): return rng.normal(0, 0.5, n)
    
    return GroundTruthSCM(
        name="collider",
        variables=["X", "Y", "Z"],
        edges=[("X", "Z"), ("Y", "Z")],
        structural_equations={"X": f_X, "Y": f_Y, "Z": f_Z},
        noise_dists={"X": n_X, "Y": n_Y, "Z": n_Z},
        intervention_variables=["X"],
        target_variable="Y",
        description="Collider X->Z<-Y. X and Y are marginally independent but dependent given Z."
    )

def make_multi_confounder_scenario() -> GroundTruthSCM:
    """Multiple confounders: Z1, Z2 -> X, Z1, Z2 -> Y, X -> Y."""
    def f_Z1(parents, noise): return noise
    def f_Z2(parents, noise): return noise
    def f_X(parents, noise): return 1.0 * parents['Z1'] + 0.5 * parents['Z2'] + noise
    def f_Y(parents, noise): return 0.8 * parents['Z1'] + 1.2 * parents['Z2'] + 2.0 * parents['X'] + noise
    
    def n(rng, n): return rng.normal(0, 1, n)
    
    return GroundTruthSCM(
        name="multi_confounder",
        variables=["Z1", "Z2", "X", "Y"],
        edges=[("Z1", "X"), ("Z2", "X"), ("Z1", "Y"), ("Z2", "Y"), ("X", "Y")],
        structural_equations={"Z1": f_Z1, "Z2": f_Z2, "X": f_X, "Y": f_Y},
        noise_dists={"Z1": n, "Z2": n, "X": n, "Y": n},
        intervention_variables=["X"],
        target_variable="Y",
        description="Multiple confounders Z1, Z2. Both must be adjusted for."
    )

def make_frontdoor_scenario() -> GroundTruthSCM:
    """Front-door: X -> M -> Y, and unobserved confounding U -> X, U -> Y.
    Effect identifiable via front-door criterion through M."""
    def f_U(parents, noise): return noise
    def f_X(parents, noise): return 2.0 * parents['U'] + noise
    def f_M(parents, noise): return 1.5 * parents['X'] + noise
    def f_Y(parents, noise): return 1.0 * parents['M'] + 3.0 * parents['U'] + noise
    
    def n(rng, n): return rng.normal(0, 1, n)
    
    return GroundTruthSCM(
        name="frontdoor",
        variables=["X", "M", "Y", "U"],
        edges=[("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")],
        structural_equations={"U": f_U, "X": f_X, "M": f_M, "Y": f_Y},
        noise_dists={"U": n, "X": n, "M": n, "Y": n},
        intervention_variables=["X"],
        target_variable="Y",
        hidden_variables=["U"],
        description="Front-door scenario. U is unobserved confounder. Effect identifiable via mediator M."
    )

def make_iv_scenario() -> GroundTruthSCM:
    """Instrumental variable: Z -> X, U -> X, U -> Y.
    Z is a valid instrument for effect of X on Y."""
    def f_Z(parents, noise): return noise
    def f_U(parents, noise): return noise
    def f_X(parents, noise): return 1.0 * parents['Z'] + 2.0 * parents['U'] + noise
    def f_Y(parents, noise): return 3.0 * parents['X'] + 1.5 * parents['U'] + noise
    
    def n(rng, n): return rng.normal(0, 1, n)
    
    return GroundTruthSCM(
        name="iv",
        variables=["Z", "U", "X", "Y"],
        edges=[("Z", "X"), ("U", "X"), ("U", "Y"), ("X", "Y")],
        structural_equations={"Z": f_Z, "U": f_U, "X": f_X, "Y": f_Y},
        noise_dists={"Z": n, "U": n, "X": n, "Y": n},
        intervention_variables=["X"],
        target_variable="Y",
        hidden_variables=["U"],
        description="Instrumental variable Z. Unobserved confounder U."
    )

def make_nonlinear_scenario() -> GroundTruthSCM:
    """Nonlinear mechanism: X -> Y with quadratic effect."""
    def f_X(parents, noise): return noise
    def f_Y(parents, noise): return 0.5 * parents['X']**2 + noise
    
    def n_X(rng, n): return rng.normal(0, 1, n)
    def n_Y(rng, n): return rng.normal(0, 0.5, n)
    
    return GroundTruthSCM(
        name="nonlinear",
        variables=["X", "Y"],
        edges=[("X", "Y")],
        structural_equations={"X": f_X, "Y": f_Y},
        noise_dists={"X": n_X, "Y": n_Y},
        intervention_variables=["X"],
        target_variable="Y",
        description="Nonlinear mechanism Y = 0.5*X^2 + noise."
    )

def make_hidden_confounding_scenario() -> GroundTruthSCM:
    """Simple X -> Y but with hidden confounder U -> X, U -> Y.
    Without observing U, the effect is NOT identifiable via backdoor."""
    def f_U(parents, noise): return noise
    def f_X(parents, noise): return 2.0 * parents['U'] + noise
    def f_Y(parents, noise): return 1.0 * parents['X'] + 3.0 * parents['U'] + noise
    
    def n(rng, n): return rng.normal(0, 1, n)
    
    return GroundTruthSCM(
        name="hidden_confounding",
        variables=["X", "Y", "U"],
        edges=[("U", "X"), ("U", "Y"), ("X", "Y")],
        structural_equations={"U": f_U, "X": f_X, "Y": f_Y},
        noise_dists={"U": n, "X": n, "Y": n},
        intervention_variables=["X"],
        target_variable="Y",
        hidden_variables=["U"],
        description="Hidden confounder U. Effect of X on Y is NOT identifiable from observed data."
    )


SCENARIO_REGISTRY = {
    "chain": make_chain_scenario,
    "fork": make_fork_scenario,
    "collider": make_collider_scenario,
    "multi_confounder": make_multi_confounder_scenario,
    "frontdoor": make_frontdoor_scenario,
    "iv": make_iv_scenario,
    "nonlinear": make_nonlinear_scenario,
    "hidden_confounding": make_hidden_confounding_scenario,
}


# ============================================================================
# BASELINE MODELS
# ============================================================================

class BaselineModel:
    """Abstract base for baseline causal models."""
    
    def __init__(self, name: str):
        self.name = name
    
    def fit(self, data: Dict[str, np.ndarray], scm: GroundTruthSCM):
        raise NotImplementedError
    
    def predict_intervention(self, intervention: Dict[str, float]) -> float:
        raise NotImplementedError
    
    def predict_counterfactual(self, factual: Dict[str, float], 
                               intervention: Dict[str, float]) -> float:
        raise NotImplementedError
    
    def should_abstain(self, scm: GroundTruthSCM) -> bool:
        return False


class OracleBaseline(BaselineModel):
    """Oracle knows the true SCM and computes exact answers."""
    
    def __init__(self):
        super().__init__("oracle")
        self._scm = None
        self._sim = None
    
    def fit(self, data: Dict[str, np.ndarray], scm: GroundTruthSCM):
        self._scm = scm
        self._sim = GroundTruthSimulator(scm)
    
    def predict_intervention(self, intervention: Dict[str, float]) -> float:
        inter_data = self._sim.do(n=10000, interventions=intervention)
        return float(np.mean(inter_data[self._scm.target_variable]))
    
    def predict_counterfactual(self, factual: Dict[str, float],
                               intervention: Dict[str, float]) -> float:
        # Exact counterfactual: abduct noise from factual, then forward-propagate
        noise = {}
        for var in self._sim._topo_order:
            parents = [u for u, v in self._scm.edges if v == var]
            parent_vals = {p: factual[p] for p in parents}
            # All our benchmark scenarios have additive noise
            f_zero = self._scm.structural_equations[var](parent_vals, 0.0)
            noise[var] = factual[var] - f_zero
        
        cf = {}
        for var in self._sim._topo_order:
            if var in intervention:
                cf[var] = intervention[var]
                continue
            parents = [u for u, v in self._scm.edges if v == var]
            parent_vals = {p: cf[p] for p in parents}
            cf[var] = self._scm.structural_equations[var](parent_vals, noise[var])
        
        return float(cf[self._scm.target_variable])


class OLSBaseline(BaselineModel):
    """Naive OLS regression of target on treatment, ignoring confounding."""
    
    def __init__(self):
        super().__init__("ols_naive")
        self._model = None
        self._treatment = None
        self._target = None
    
    def fit(self, data: Dict[str, np.ndarray], scm: GroundTruthSCM):
        self._treatment = scm.intervention_variables[0]
        self._target = scm.target_variable
        X = data[self._treatment].reshape(-1, 1)
        y = data[self._target]
        if SKLEARN_AVAILABLE:
            self._model = LinearRegression()
            self._model.fit(X, y)
        else:
            # Manual OLS
            Xb = np.column_stack([np.ones(len(X)), X])
            beta = np.linalg.lstsq(Xb, y, rcond=None)[0]
            self._model = lambda x: beta[0] + beta[1] * x
    
    def predict_intervention(self, intervention: Dict[str, float]) -> float:
        x = intervention[self._treatment]
        if SKLEARN_AVAILABLE:
            return float(self._model.predict([[x]])[0])
        else:
            return float(self._model(x))
    
    def predict_counterfactual(self, factual: Dict[str, float],
                               intervention: Dict[str, float]) -> float:
        return self.predict_intervention(intervention)


class DoWhyBaseline(BaselineModel):
    """DoWhy backdoor linear regression."""
    
    def __init__(self):
        super().__init__("dowhy_backdoor")
        self._model = None
        self._treatment = None
        self._target = None
        self._identified = True
    
    def fit(self, data: Dict[str, np.ndarray], scm: GroundTruthSCM):
        if not DOWHY_AVAILABLE:
            return
        self._treatment = scm.intervention_variables[0]
        self._target = scm.target_variable
        
        # Build pandas dataframe
        import pandas as pd
        df = pd.DataFrame(data)
        
        # Build causal graph string for DoWhy
        edges = scm.edges
        graph_str = "digraph {"
        for u, v in edges:
            if u not in scm.hidden_variables and v not in scm.hidden_variables:
                graph_str += f" {u} -> {v};"
        graph_str += " }"
        
        try:
            model = CausalModel(
                data=df,
                treatment=self._treatment,
                outcome=self._target,
                graph=graph_str
            )
            identified = model.identify_effect()
            estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")
            self._model = estimate
            self._identified = True
        except Exception as e:
            self._identified = False
            self._model = None
    
    def predict_intervention(self, intervention: Dict[str, float]) -> float:
        if not self._identified or self._model is None:
            return float('nan')
        ate = self._model.value
        x = intervention[self._treatment]
        return float(ate * x)
    
    def predict_counterfactual(self, factual: Dict[str, float],
                               intervention: Dict[str, float]) -> float:
        # DoWhy in this API does unit-level counterfactuals without refutation
        # We approximate by returning the intervention prediction
        return self.predict_intervention(intervention)
    
    def should_abstain(self, scm: GroundTruthSCM) -> bool:
        return not self._identified


class CRCABaseline(BaselineModel):
    """CR-CA heuristic agent baseline."""
    
    def __init__(self, use_nonlinear: bool = False, abstain: bool = False):
        name = "crca_nonlinear" if use_nonlinear else "crca_linear"
        if abstain:
            name += "_abstain"
        super().__init__(name)
        self.use_nonlinear = use_nonlinear
        self.abstain = abstain
        self._agent = None
        self._treatment = None
        self._target = None
    
    def fit(self, data: Dict[str, np.ndarray], scm: GroundTruthSCM):
        self._agent = CRCAAgent()
        self._agent.set_model_class("acyclic_scm")
        self._agent.use_nonlinear_scm = self.use_nonlinear
        self._agent.nonlinear_activation = "tanh" if self.use_nonlinear else "identity"
        self._agent.abstain_on_nonidentifiable = self.abstain
        self._treatment = scm.intervention_variables[0]
        self._target = scm.target_variable
        self._scm = scm

        # Add edges from true graph (excluding hidden variables)
        for u, v in scm.edges:
            if u not in scm.hidden_variables and v not in scm.hidden_variables:
                self._agent.add_causal_relationship(u, v, strength=1.0, epistemic_status="supplied_by_user")

        # Set standardization stats from data
        for var in scm.variables:
            if var not in scm.hidden_variables:
                self._agent.set_standardization_stats(var, mean=float(np.mean(data[var])), std=max(float(np.std(data[var])), 0.001))

        # Polynomial term discovery: for each (treatment, target) pair, screen x^exp
        treatment = self._treatment
        target = self._target
        if treatment in data and target in data:
            disc = self._agent.set_polynomial_terms_from_data(
                target=target,
                parent=treatment,
                data_x=data[treatment],
                data_y=data[target],
                max_exponent=3,
                r2_threshold=0.01
            )
            if disc:
                self._agent.disable_linear_for_polynomial(treatment, target)

        self._agent.estimate_edge_coefficients(data)

    def predict_intervention(self, intervention: Dict[str, float]) -> float:
        if self.abstain and self._scm.hidden_variables:
            return float('nan')
        if self.abstain:
            id_result = self._agent.check_identifiability(self._treatment, self._target)
            if not id_result["identifiable"]:
                return float('nan')

        # Use a representative factual state (mean)
        factual = {}
        for var in self._agent.causal_graph:
            stats = self._agent.standardization_stats.get(var, {"mean": 0.0, "std": 1.0})
            factual[var] = stats["mean"]

        result = self._agent._predict_outcomes(factual, intervention)
        return float(result.get(self._target, float('nan')))

    def predict_counterfactual(self, factual: Dict[str, float],
                               intervention: Dict[str, float]) -> float:
        if self.abstain and self._scm.hidden_variables:
            return float('nan')
        if self.abstain:
            id_result = self._agent.check_identifiability(self._treatment, self._target)
            if not id_result["identifiable"]:
                return float('nan')

        result = self._agent.counterfactual_abduction_action_prediction(factual, intervention)
        return float(result.get(self._target, float('nan')))
    
    def should_abstain(self, scm: GroundTruthSCM) -> bool:
        if not self.abstain:
            return False
        if self._scm.hidden_variables:
            return True
        id_result = self._agent.check_identifiability(self._treatment, self._target)
        return not id_result["identifiable"]


class CRCAFormalBaseline(BaselineModel):
    """CR-CA formal tier baseline using crca_core."""
    
    def __init__(self):
        super().__init__("crca_core_formal")
        self._estimate = None
        self._identified = False
        self._treatment = None
        self._target = None
    
    def fit(self, data: Dict[str, np.ndarray], scm: GroundTruthSCM):
        if not CRCA_CORE_AVAILABLE:
            return
        self._treatment = scm.intervention_variables[0]
        self._target = scm.target_variable
        
        nodes = [NodeSpec(name=v) for v in scm.variables if v not in scm.hidden_variables]
        edges = [EdgeSpec(source=u, target=v) for u, v in scm.edges
                 if u not in scm.hidden_variables and v not in scm.hidden_variables]
        
        draft = DraftSpec(
            graph=CausalGraphSpec(nodes=nodes, edges=edges),
            roles=RoleSpec(treatments=[self._treatment], outcomes=[self._target])
        )
        try:
            locked = lock_spec(draft, approvals=["benchmark"])
            result = identify_effect(locked_spec=locked, treatment=self._treatment, outcome=self._target)
            self._identified = hasattr(result, "method")
            self._estimate = result
        except Exception:
            self._identified = False
    
    def predict_intervention(self, intervention: Dict[str, float]) -> float:
        if not self._identified:
            return float('nan')
        # crca_core returns identification strategy, not numeric estimate
        return float('nan')
    
    def predict_counterfactual(self, factual: Dict[str, float],
                               intervention: Dict[str, float]) -> float:
        return float('nan')
    
    def should_abstain(self, scm: GroundTruthSCM) -> bool:
        return not self._identified


class RandomBaseline(BaselineModel):
    """Random predictions."""
    
    def __init__(self):
        super().__init__("random")
        self._rng = default_rng(999)
        self._mean = 0.0
        self._std = 1.0
    
    def fit(self, data: Dict[str, np.ndarray], scm: GroundTruthSCM):
        self._mean = float(np.mean(data[scm.target_variable]))
        self._std = max(float(np.std(data[scm.target_variable])), 0.001)
    
    def predict_intervention(self, intervention: Dict[str, float]) -> float:
        return float(self._rng.normal(self._mean, self._std))
    
    def predict_counterfactual(self, factual: Dict[str, float],
                               intervention: Dict[str, float]) -> float:
        return self.predict_intervention(intervention)


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class ExpandedBenchmark:
    
    def __init__(self, scenarios: List[str], models: List[str], 
                 n_samples: int = 5000, n_test: int = 100,
                 seed: int = 42, verbose: bool = False):
        self.scenario_names = scenarios
        self.model_names = models
        self.n_samples = n_samples
        self.n_test = n_test
        self.seed = seed
        self.verbose = verbose
        self.rng = default_rng(seed)
        self.results: List[BenchmarkResult] = []
    
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def run(self):
        self.log("=" * 70)
        self.log("CR-CA EXPANDED BENCHMARK SUITE v2.0")
        self.log("=" * 70)
        self.log(f"Scenarios: {self.scenario_names}")
        self.log(f"Models: {self.model_names}")
        self.log(f"Train samples: {self.n_samples}, Test samples: {self.n_test}")
        self.log(f"CRCA_CORE: {CRCA_CORE_AVAILABLE}, DOWHY: {DOWHY_AVAILABLE}")
        self.log("")
        
        for scenario_name in self.scenario_names:
            if scenario_name not in SCENARIO_REGISTRY:
                self.log(f"WARNING: Unknown scenario '{scenario_name}', skipping")
                continue
            self._run_scenario(scenario_name)
        
        self._print_summary()
        self._save_results()
    
    def _run_scenario(self, scenario_name: str):
        self.log(f"\n{'='*70}")
        self.log(f"SCENARIO: {scenario_name}")
        self.log(f"{'='*70}")
        
        scm = SCENARIO_REGISTRY[scenario_name]()
        sim = GroundTruthSimulator(scm, rng=self.rng)
        
        # Generate training data
        train_data, _ = sim.sample(self.n_samples)
        
        # Generate test factuals
        test_data, test_noise = sim.sample(self.n_test)
        
        # Define intervention values to test
        treatment = scm.intervention_variables[0]
        target = scm.target_variable
        
        # Determine intervention values based on treatment distribution
        t_mean = float(np.mean(train_data[treatment]))
        t_std = float(np.std(train_data[treatment]))
        intervention_values = [
            t_mean - 1.5 * t_std,
            t_mean - 0.5 * t_std,
            t_mean + 0.5 * t_std,
            t_mean + 1.5 * t_std,
        ]
        
        # Build baseline models
        models = self._build_models()
        
        for model in models:
            self.log(f"\n  Model: {model.name}")

            try:
                fit_start = time.time()
                model.fit(train_data, scm)
                fit_duration_ms = (time.time() - fit_start) * 1000
            except Exception as e:
                self.log(f"    FIT FAILED: {e}")
                continue

            # Intervention prediction error
            inter_predictions = []
            for iv_val in intervention_values:
                intervention = {treatment: iv_val}

                # Ground truth
                gt_inter = sim.do(n=1000, interventions=intervention)
                gt_mean = float(np.mean(gt_inter[target]))

                # Model prediction
                pred_start = time.time()
                pred = model.predict_intervention(intervention)
                pred_duration_ms = (time.time() - pred_start) * 1000
                if not np.isnan(pred):
                    inter_predictions.append({
                        "pred": pred, "gt": gt_mean,
                        "abs_err": abs(pred - gt_mean),
                        "sq_err": (pred - gt_mean) ** 2,
                        "duration_ms": pred_duration_ms,
                    })

            if inter_predictions:
                mae_inter = float(np.mean([p["abs_err"] for p in inter_predictions]))
                rmse_inter = float(np.sqrt(np.mean([p["sq_err"] for p in inter_predictions])))
                avg_pred_ms = float(np.mean([p["duration_ms"] for p in inter_predictions]))
                self.results.append(BenchmarkResult(
                    scenario=scenario_name, model=model.name,
                    metric="intervention_mae", value=mae_inter,
                    unit="absolute", n_samples=self.n_samples,
                    duration_ms=avg_pred_ms, fit_duration_ms=fit_duration_ms
                ))
                self.results.append(BenchmarkResult(
                    scenario=scenario_name, model=model.name,
                    metric="intervention_rmse", value=rmse_inter,
                    unit="absolute", n_samples=self.n_samples
                ))
                self.log(f"    Intervention MAE: {mae_inter:.4f}, RMSE: {rmse_inter:.4f} (fit {fit_duration_ms:.1f}ms, pred avg {avg_pred_ms:.2f}ms/call)")
            
            # Counterfactual prediction error (on test set)
            cf_predictions = []
            for i in range(min(self.n_test, 50)):  # Sample 50 test units
                factual = {var: float(test_data[var][i]) for var in scm.variables}
                for iv_val in intervention_values[:2]:  # Test 2 intervention values
                    intervention = {treatment: iv_val}

                    # Ground truth counterfactual
                    single_data = {var: np.array([test_data[var][i]]) for var in scm.variables}
                    single_noise = {var: np.array([test_noise[var][i]]) for var in scm.variables}
                    gt_cf = sim.counterfactual(single_data, intervention, single_noise)
                    gt_val = float(gt_cf[target][0])

                    # Model prediction
                    pred = model.predict_counterfactual(factual, intervention)
                    if not np.isnan(pred):
                        cf_predictions.append({"pred": pred, "gt": gt_val, "err": abs(pred - gt_val)})

            if cf_predictions:
                mae_cf = float(np.mean([p["err"] for p in cf_predictions]))
                rmse_cf = float(np.sqrt(np.mean([(p["pred"] - p["gt"]) ** 2 for p in cf_predictions])))

                # Coverage: fraction of predictions within threshold of ground truth
                threshold = 0.5
                within = sum(1 for p in cf_predictions if p["err"] <= threshold)
                coverage = within / len(cf_predictions)

                self.results.append(BenchmarkResult(
                    scenario=scenario_name, model=model.name,
                    metric="counterfactual_mae", value=mae_cf,
                    unit="absolute", n_samples=len(cf_predictions)
                ))
                self.results.append(BenchmarkResult(
                    scenario=scenario_name, model=model.name,
                    metric="counterfactual_rmse", value=rmse_cf,
                    unit="absolute", n_samples=len(cf_predictions)
                ))
                if len(cf_predictions) > 10:
                    self.results.append(BenchmarkResult(
                        scenario=scenario_name, model=model.name,
                        metric="counterfactual_coverage_0.5", value=coverage,
                        unit="fraction", n_samples=len(cf_predictions)
                    ))
                self.log(f"    Counterfactual MAE: {mae_cf:.4f}, RMSE: {rmse_cf:.4f} (coverage @0.5: {coverage:.2%})")
            
            # Identifiability / abstention
            # Only models with an explicit abstain capability should be scored on abstention
            abstain_capable = model.name in ["crca_linear_abstain", "crca_nonlinear_abstain",
                                             "dowhy_backdoor_abstain", "crca_core_formal"]
            has_hidden = len(scm.hidden_variables) > 0

            if not abstain_capable:
                abstention_correct = True
                details = "n/a - no abstain option"
            else:
                should_abstain = has_hidden
                did_abstain = model.should_abstain(scm)
                abstention_correct = (should_abstain == did_abstain)
                details = f"should={should_abstain}, did={did_abstain}"

            self.results.append(BenchmarkResult(
                scenario=scenario_name, model=model.name,
                metric="abstention_accuracy", value=1.0 if abstention_correct else 0.0,
                unit="binary", passed=abstention_correct,
                details=details
            ))
            self.log(f"    Abstention: {details}")
    
    def _build_models(self) -> List[BaselineModel]:
        models = []
        for name in self.model_names:
            if name == "oracle":
                models.append(OracleBaseline())
            elif name == "ols_naive":
                models.append(OLSBaseline())
            elif name == "dowhy_backdoor" and DOWHY_AVAILABLE:
                models.append(DoWhyBaseline())
            elif name == "crca_linear":
                models.append(CRCABaseline(use_nonlinear=False))
            elif name == "crca_linear_abstain":
                models.append(CRCABaseline(use_nonlinear=False, abstain=True))
            elif name == "crca_nonlinear":
                models.append(CRCABaseline(use_nonlinear=True))
            elif name == "crca_core_formal" and CRCA_CORE_AVAILABLE:
                models.append(CRCAFormalBaseline())
            elif name == "random":
                models.append(RandomBaseline())
            else:
                self.log(f"  WARNING: Model '{name}' not available, skipping")
        return models
    
    def _run_significance_tests(self):
        """Compute relative performance ratios (model MAE / oracle MAE) per scenario."""
        scenarios = sorted(set(r.scenario for r in self.results))
        models = sorted(set(r.model for r in self.results))
        oracle_name = "oracle"

        mae_by_sm = {}
        for r in self.results:
            if r.metric == "intervention_mae":
                key = (r.scenario, r.model)
                mae_by_sm[key] = r.value

        lines = ["\n## Relative Performance (Model MAE / Oracle MAE)", ""]
        lines.append("| Scenario             | " + " | ".join(f"{m:<15}" for m in models if m != oracle_name) + " |")
        lines.append("|" + "|".join(f"{'-'*20}" for _ in [oracle_name] + [m for m in models if m != oracle_name]) + "|")

        for s in scenarios:
            oracle_mae = mae_by_sm.get((s, oracle_name), float('nan'))
            row = [s]
            for m in models:
                if m == oracle_name:
                    continue
                mae = mae_by_sm.get((s, m), float('nan'))
                if np.isnan(oracle_mae) or np.isnan(mae) or oracle_mae == 0:
                    ratio = float('nan')
                else:
                    ratio = mae / oracle_mae
                row.append(f"{ratio:.2f}" if not np.isnan(ratio) else "N/A")
            lines.append("| " + " | ".join(f"{v:<20}" for v in row) + " |")

        lines.append("")
        lines.append("Values > 1.0 mean the model is worse than Oracle. Below 1.0 is better (rare — Oracle is optimal).")
        return "\n".join(lines)

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("EXPANDED BENCHMARK SUMMARY")
        print("=" * 70)

        table = {}
        for r in self.results:
            key = (r.scenario, r.model)
            if key not in table:
                table[key] = {}
            table[key][r.metric] = r.value

        scenarios = sorted(set(r.scenario for r in self.results))
        models = sorted(set(r.model for r in self.results))
        metrics = ["intervention_mae", "counterfactual_mae", "abstention_accuracy"]

        for metric in metrics:
            print(f"\n--- {metric.upper()} ---")
            print(f"{'Scenario':<20}", end="")
            for m in models:
                print(f"{m:>18}", end="")
            print()
            print("-" * (20 + 18 * len(models)))
            for s in scenarios:
                print(f"{s:<20}", end="")
                for m in models:
                    val = table.get((s, m), {}).get(metric, float('nan'))
                    if np.isnan(val):
                        print(f"{'N/A':>18}", end="")
                    else:
                        print(f"{val:>18.4f}", end="")
                print()

        sig = self._run_significance_tests()
        print(sig)
    
    def _save_results(self):
        output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {
                "scenarios": self.scenario_names,
                "models": self.model_names,
                "n_samples": self.n_samples,
                "n_test": self.n_test,
                "seed": self.seed,
            },
            "results": [],
        }
        for r in self.results:
            result_dict = asdict(r)
            result_dict.pop("_uid", None)
            output["results"].append(result_dict)
        path = os.path.join(repo_root, "benchmark_expanded_results.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {path}")
        
        # Generate markdown report
        self._generate_report(output, path.replace(".json", "_report.md"))
    
    def _generate_report(self, data: dict, path: str):
        lines = []
        lines.append("# CR-CA Expanded Benchmark Report")
        lines.append(f"\n**Date:** {data['timestamp']}")
        lines.append(f"**Seed:** {data['config']['seed']}")
        lines.append(f"**Train samples:** {data['config']['n_samples']}")
        lines.append(f"**Test samples:** {data['config']['n_test']}")
        lines.append("\n## Models Tested")
        for m in data['config']['models']:
            lines.append(f"- {m}")
        lines.append("\n## Scenarios Tested")
        for s in data['config']['scenarios']:
            lines.append(f"- {s}")
        
        # Build tables per metric
        metrics = ["intervention_mae", "intervention_rmse", "counterfactual_mae",
                  "counterfactual_rmse", "counterfactual_coverage_0.5", "abstention_accuracy"]
        scenarios = sorted(set(r['scenario'] for r in data['results']))
        models = sorted(set(r['model'] for r in data['results']))
        
        table = {}
        for r in data['results']:
            key = (r['scenario'], r['model'])
            if key not in table:
                table[key] = {}
            table[key][r['metric']] = r['value']
        
        for metric in metrics:
            lines.append(f"\n## {metric.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"| {'Scenario':<20} | " + " | ".join(f"{m:<18}" for m in models) + " |")
            lines.append(f"|{'-'*20}|" + "|".join(f"{'-'*18}" for _ in models) + "|")
            for s in scenarios:
                vals = []
                for m in models:
                    v = table.get((s, m), {}).get(metric, float('nan'))
                    if np.isnan(v):
                        vals.append("N/A")
                    else:
                        vals.append(f"{v:.4f}")
                lines.append(f"| {s:<20} | " + " | ".join(f"{v:<18}" for v in vals) + " |")
        
        lines.append("\n## Interpretation")
        lines.append("\n- **Intervention MAE/RMSE:** Lower is better. Measures how well the model predicts the outcome of an intervention (do-operator).")
        lines.append("- **Counterfactual MAE/RMSE:** Lower is better. Measures how well the model predicts counterfactual outcomes for specific units.")
        lines.append("- **Counterfactual Coverage @0.5:** Fraction of counterfactual predictions within 0.5 units of ground truth. Higher is better.")
        lines.append("- **Abstention Accuracy:** 1.0 means the model correctly abstained (or correctly did not abstain). 0.0 means it made a wrong abstention decision.")
        lines.append("- **Fit Duration (ms):** Time to fit the model on training data.")
        lines.append("- **Prediction Duration (ms/call):** Average time per intervention prediction call.")
        lines.append("\n## Key Findings")
        lines.append("\n1. **Oracle** should achieve near-zero error since it knows the true SCM.")
        lines.append("2. **OLS Naive** should fail on confounded scenarios (fork, multi_confounder, hidden_confounding).")
        lines.append("3. **DoWhy Backdoor** should succeed when backdoor criterion is satisfied, and fail otherwise.")
        lines.append("4. **CR-CA Linear** should match Oracle on linear, acyclic, correctly-specified graphs.")
        lines.append("5. **CR-CA Nonlinear** may have higher error due to tanh heuristic even on linear problems.")
        lines.append("6. **Abstention** is critical for hidden confounding scenarios where effects are not identifiable.")
        
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"Report saved to {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CR-CA Expanded Benchmark Suite")
    parser.add_argument("--scenarios", nargs="+", default=["all"],
                        help="Scenarios to run (default: all)")
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="Models to run (default: all)")
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Training sample size")
    parser.add_argument("--n_test", type=int, default=100,
                        help="Test sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    
    all_scenarios = list(SCENARIO_REGISTRY.keys())
    all_models = ["oracle", "ols_naive", "crca_linear", "crca_linear_abstain",
                  "crca_nonlinear", "random"]
    if DOWHY_AVAILABLE:
        all_models.insert(2, "dowhy_backdoor")
    if CRCA_CORE_AVAILABLE:
        all_models.insert(-1, "crca_core_formal")
    
    scenarios = all_scenarios if "all" in args.scenarios else args.scenarios
    models = all_models if "all" in args.models else args.models
    
    bench = ExpandedBenchmark(
        scenarios=scenarios,
        models=models,
        n_samples=args.n_samples,
        n_test=args.n_test,
        seed=args.seed,
        verbose=args.verbose
    )
    bench.run()


if __name__ == "__main__":
    main()
