# CR-CA Expanded Benchmark Report

**Date:** 2026-05-27T23:08:21
**Seed:** 42
**Train samples:** 5000
**Test samples:** 100

## Models Tested
- crca_linear
- ols_naive
- oracle

## Scenarios Tested
- collider
- chain

## Intervention Mae

| Scenario             | crca_linear        | ols_naive          | oracle             |
|--------------------|------------------|------------------|------------------|
| chain                | 0.0107             | 0.0220             | 0.0190             |
| collider             | 0.0440             | 0.0351             | 0.0324             |

## Intervention Rmse

| Scenario             | crca_linear        | ols_naive          | oracle             |
|--------------------|------------------|------------------|------------------|
| chain                | 0.0114             | 0.0258             | 0.0210             |
| collider             | 0.0635             | 0.0454             | 0.0370             |

## Counterfactual Mae

| Scenario             | crca_linear        | ols_naive          | oracle             |
|--------------------|------------------|------------------|------------------|
| chain                | 0.0019             | 0.4024             | 0.0000             |
| collider             | 0.0000             | 0.8709             | 0.0000             |

## Counterfactual Rmse

| Scenario             | crca_linear        | ols_naive          | oracle             |
|--------------------|------------------|------------------|------------------|
| chain                | 0.0023             | 0.5190             | 0.0000             |
| collider             | 0.0000             | 1.0277             | 0.0000             |

## Counterfactual Coverage 0.5

| Scenario             | crca_linear        | ols_naive          | oracle             |
|--------------------|------------------|------------------|------------------|
| chain                | 1.0000             | 0.6400             | 1.0000             |
| collider             | 1.0000             | 0.2700             | 1.0000             |

## Abstention Accuracy

| Scenario             | crca_linear        | ols_naive          | oracle             |
|--------------------|------------------|------------------|------------------|
| chain                | 1.0000             | 1.0000             | 1.0000             |
| collider             | 1.0000             | 1.0000             | 1.0000             |

## Interpretation

- **Intervention MAE/RMSE:** Lower is better. Measures how well the model predicts the outcome of an intervention (do-operator).
- **Counterfactual MAE/RMSE:** Lower is better. Measures how well the model predicts counterfactual outcomes for specific units.
- **Counterfactual Coverage @0.5:** Fraction of counterfactual predictions within 0.5 units of ground truth. Higher is better.
- **Abstention Accuracy:** 1.0 means the model correctly abstained (or correctly did not abstain). 0.0 means it made a wrong abstention decision.
- **Fit Duration (ms):** Time to fit the model on training data.
- **Prediction Duration (ms/call):** Average time per intervention prediction call.

## Key Findings

1. **Oracle** should achieve near-zero error since it knows the true SCM.
2. **OLS Naive** should fail on confounded scenarios (fork, multi_confounder, hidden_confounding).
3. **DoWhy Backdoor** should succeed when backdoor criterion is satisfied, and fail otherwise.
4. **CR-CA Linear** should match Oracle on linear, acyclic, correctly-specified graphs.
5. **CR-CA Nonlinear** may have higher error due to tanh heuristic even on linear problems.
6. **Abstention** is critical for hidden confounding scenarios where effects are not identifiable.