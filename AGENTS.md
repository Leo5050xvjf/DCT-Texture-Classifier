# Codex hand-off instructions

Read `EXPERIMENT_STATE.md` first. It is the canonical current state and takes
precedence over older DCT-era wording in historical reports.

Terminology:

- `P` = spatial 32x32 patch texture CNN (formerly STCNN/Spatial32).
- `G` = dense residual U-Net-like texture-map generator.
- The current main line is P -> clean dense teacher map -> G.
- DCT is a historical baseline unless a request explicitly concerns it.

Important cautions:

- Do not call one P globally best. Distinguish P-clean-teacher, P-robust, and
  P-distilled by role.
- Patch accuracy, dense self-consistency, and G-to-teacher agreement are
  different metrics and must not be compared as if identical.
- The external grass suite has been inspected during selection and is not a
  blind final test set. `div2k_0524` is a training image.
- Data and generated NPZ/NPY files are intentionally ignored. Do not claim they
  are on GitHub; use the documented rebuild commands.
- Do not open or visually inspect result images unless the user explicitly asks.
  Prefer JSON/CSV/manifests for routine status and analysis.
- Preserve experiment artifacts and unrelated user changes. Long training or
  inference jobs should write logs and run through the provided background BAT
  launchers so conversation is not blocked by frequent progress output.

Before reporting a new result, record the exact checkpoint, split, corruption,
seed, threshold, and metric definition. Update `EXPERIMENT_STATE.md` when the
recommended model or main conclusion changes.
