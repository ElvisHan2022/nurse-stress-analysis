# Timezone resolution

**Resolved:** survey timestamps are naive **America/Chicago** (DST-aware).
Sensor `t0` values are UTC unix timestamps.

Evidence, on 245 rated events:

| Hypothesis | Events whose start falls inside a same-subject session |
|---|---|
| Survey timestamps are UTC | 50 (20.4%) |
| Survey naive America/Chicago -> UTC | **212 (86.5%)** |

Best fixed offset was UTC-5 at 198
events. DST-aware `America/Chicago` reaches 212, beating it by
14. The archive spans 2020-04 to 2020-12,
so a fixed offset misaligns winter events by an hour. **Use the named zone.**

Residual: 33 rated events (13.5%) still fall outside
every session under the winning hypothesis. These are survey reports with no
sensor coverage, not a timezone failure.

Every later section uses this resolution.
