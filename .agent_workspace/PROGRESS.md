# PROGRESS — femtools 1:1 SOTA

## Branch

`cursor/femtools-sota-d551`

## Goal

Implement an original, SOTA-quality functional equivalent of FEMtools (Dynamic Design Solutions): Framework, Dynamics, Pretest & Correlation, Model Updating, Optimization, MPE, RBPE, FEA/test I/O, scripting, CLI/GUI.

## Loop status

| Round | Status | Notes |
|-------|--------|--------|
| Prep | done | Contract API, ownership, package seed |
| Round 1 | in_flight | 3 cloud VM (R1-F1/F2/F3) + 7 local exclusive-path agents (platform async new-VM cap = 3) |
| Round 2 | pending | |
| Round 3 | pending | |

## Round 1 dispatch map

See `FILE_OWNERSHIP.md`.

Cloud VM (base `cursor/femtools-sota-d551`):
- R1-F1 `bc-1607c7fa-f177-5b9f-bea9-77289bf37d34` claude-fable-5-thinking-xhigh
- R1-F2 `bc-6f6a7d5a-4380-5cfe-a001-7d77103402de` claude-fable-5-thinking-xhigh
- R1-F3 `bc-44f88846-aa6d-5e5c-98c8-ff1ec5447d96` claude-fable-5-thinking-xhigh

Local exclusive-path (same branch; agents must NOT git commit/push):
- R1-F4 fable CLI/GUI/script
- R1-O1..O4 opus-fast implementation
- R1-G1..G2 gpt-sol tests/benchmarks
