# Proposed research contribution

Working title: **Reducing Identity Errors in Personalized Visual Object Memory**.

Research question: can reference guidance, selective confirmation and scoped historical revision reduce wrong personal-object memories while retaining useful memory coverage and limiting review workload?

Related work already includes teachable personal recognition (Find My Things) and streaming visual object memory (ESOM). Uncertainty, active learning and correction are not globally new concepts. This package provides a candidate method and evaluation workflow; a literature comparison is still required before claiming novelty.

## Implemented hypothesis

1. Compare registered objects as hard negatives. Recommend more distinguishing views for confusing pairs and select reference views using distinctiveness plus diversity.
2. Use per-item support-derived thresholds/margins and temporal agreement. Automatically accept only stable separated candidates; keep ambiguous observations tentative and surface a throttled review request.
3. Store sampled dated evidence and separate tentative observations from accepted identity memory. Explicitly review selected historical observations, invalidate unsupported identities rather than blindly relabelling them, retain before/after audits, and rebuild affected last-seen records.

The reference thresholds are heuristics, not calibrated risk guarantees. Scoped repair uses stored scores or explicit user inspection; it does not implement a learned backward identity graph or automatic historical re-embedding. Those remain potential extensions.

## Fair experiments

Hold detector and appearance model fixed. Compare nearest/latest-guess updates, fixed-threshold temporal updates, always-review, and selective policy. Ablate uniform versus guided references under fixed thresholds, then add adaptive thresholds, then add history revision. Use equal reference and review budgets. Keep whole recording sessions separated; never put neighbouring frames from one clip in reference and test sets to inflate results.

Measure wrong accepted identities/locations, update/search coverage, last-seen time error, review requests and actual user decisions, post-correction errors, track switches, CPU latency and memory/storage cost. User-assisted and automatic outcomes need separate counts. Low error from refusing nearly all updates is not success.

The public-image tool evaluates only appearance matching. A memory-repair claim also needs real chronological recordings with independently labelled identities and query times. ORBIT data may lack every required multi-object crossing/removal scenario. Report that limitation instead of fabricating a sequence.

## Report structure

Problem; related work; precise candidate contribution; architecture; algorithms/heuristics; dataset and splits; experimental setup; baselines/ablations; measured results; failure cases; limitations; reproducibility; conclusion; sources.

Useful primary sources:

- Find My Things: https://www.microsoft.com/en-us/research/wp-content/uploads/2024/06/Morrison-Find-My-Things-ASSETS2023.pdf
- ESOM: https://arxiv.org/abs/2411.16934
- ORBIT: https://github.com/microsoft/ORBIT-Dataset
- ORBIT-India: https://zenodo.org/records/12608444

Do not claim a world-first system or perfect detection. Report empirical improvement only after executing the evaluation.
