# ExperimentLab demo

All data are synthetic.

- healthy: **SHIP**; naive effect 0.462, p=1.67e-15; CUPED effect 0.381; SRM p=0.423.
- broken: **INVALID_RANDOMISATION**; naive effect 1.629, p=2.27e-162; CUPED effect -0.027; SRM p=2.86e-109.
- early: **CONTINUE_TO_FIXED_HORIZON**; naive effect 0.086, p=0.587; CUPED effect 0.425; SRM p=0.396.

The broken experiment is not a causal estimate: selective treatment logging invalidates randomisation. CUPED does not repair selection bias. Repeated-look Bonferroni is a conservative finite-look safeguard, not an anytime-valid procedure.
