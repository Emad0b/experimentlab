# ExperimentLab: methodology

The seeded healthy experiment has independent users, 50:50 assignment, a continuous outcome correlated with pre-period activity, and an additive treatment effect. A separate synthetic pilot estimates the CUPED coefficient. The broken experiment has zero true effect, but drops low-baseline treatment users after assignment. This creates an apparent win and a sample-ratio mismatch. The decision gate blocks launch even if the naive effect is significant. CUPED is a precision tool, not a repair for selection bias. The demo horizon is 2,000 users per arm; planning reports both raw and anticipated CUPED requirements, with the latter depending on the independent pilot assumption.

## Assumptions and limits

SRM does not catch all assignment errors: balanced but biased assignments can pass. Baseline imbalance is diagnostic rather than automatic proof of bias. Welch intervals assume independent randomisation units and sufficiently stable moments; clustered experiments need cluster-level analysis. The primary outcome is continuous, not a conversion-specific binary estimator. The peeking safeguard assumes ten prespecified looks and is not anytime-valid. Do not repeatedly add looks or extend an experiment after a disappointing final result. Subgroup results are exploratory; BH controls FDR under its dependence assumptions, not the probability of any false discovery.

## Extension plan

Add cluster randomisation, explicit missingness contracts, binary/ratio metrics, randomisation inference and a genuine sequential-testing design.

Planning distinction: the sample-size utility targets rejection of a zero-effect null at the specified detectable effect. The demo launch policy is stricter: its lower confidence bound must exceed 0.15. The planning figures therefore are not a claimed 80% probability of satisfying the launch rule at a true effect of 0.15.
