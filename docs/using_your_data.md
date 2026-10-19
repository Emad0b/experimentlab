# Reusable API

```python
from experimentlab.core import analyse
# Equal-length user-level arrays. assignment: 0/1. theta: independent pilot.
result = analyse(pre, outcome, assignment, guardrail, theta=pilot_theta)
```

Validate units, timestamps, missingness and evaluation assumptions. The simulator is not a substitute for representative labelled data.
