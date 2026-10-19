from .provenance import capture
import argparse
import json
from pathlib import Path
import numpy as np
from .core import (
    analyse,
    simulate,
    sample_size,
    bh_adjust,
    peeking_simulation,
    subgroup_analysis,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Independent pilot estimates theta; no outcomes from the experiment used to fit it.
    rng = np.random.default_rng(12)
    pilot_x = rng.normal(size=10000)
    pilot_y = 2 * pilot_x + rng.normal(size=10000)
    theta = float(np.cov(pilot_x, pilot_y, ddof=1)[0, 1] / np.var(pilot_x, ddof=1))
    report = {
        "data": "SYNTHETIC; independent units; one primary outcome",
        "pilot_theta": theta,
        "planning": {
            "raw_per_arm": sample_size(sd=np.sqrt(5), mde=0.15),
            "anticipated_cuped_per_arm": sample_size(sd=1, mde=0.15),
            "demo_fixed_horizon_per_arm": 2000,
            "mde": 0.15,
            "power": 0.8,
        },
        "healthy": analyse(*simulate(), theta=theta),
        "broken": analyse(*simulate(effect=0, broken=True), theta=theta),
        "early": analyse(*simulate(n=800), theta=theta),
        "peeking": peeking_simulation(),
        "exploratory_subgroups": subgroup_analysis(*simulate(), theta=theta),
        "exploratory_bh_example": bh_adjust([0.001, 0.03, 0.04, 0.8]),
    }
    (out / "results.json").write_text(json.dumps(report, indent=2))
    text = "# ExperimentLab demo\n\nAll data are synthetic.\n\n"
    for key in ["healthy", "broken", "early"]:
        r = report[key]
        text += f"- {key}: **{r['decision']}**; naive effect {r['raw']['effect']:.3f}, p={r['raw']['p']:.3g}; CUPED effect {r['cuped']['effect']:.3f}; SRM p={r['srm_p']:.3g}.\n"
    text += "\nThe broken experiment is not a causal estimate: selective treatment logging invalidates randomisation. CUPED does not repair selection bias. Repeated-look Bonferroni is a conservative finite-look safeguard, not an anytime-valid procedure.\n"
    (out / "summary.md").write_text(text)
    print(text)
    capture(out)


if __name__ == "__main__":
    main()
