# NRP - New Rebellion Paper
This is the repository of a scholarly research paper on the causes of the 1536 Pilgrimage of Grace, a rebellion in Northern England. I am seeking to determine whether the impact of the Dissolution of the Monasteries on ordinary people, the disloyalty of local elites, or other factors drove the rebellion using quantitative analysis.

## Priorities (descending)
1. Data integrity  2. Reproducibility  3. Transparency  4. Precision  5. Speed

Transparency > speed. Control > autonomy.

## Rules
- Read files before editing. Do only what is asked.
- Simple and explicit over clever.
- Code should be pythonic and human-readable. Structure your code appropriately, divide Jupyter Notebooks into cells with clear functions, and comment the code extensively.
- Don't refactor, rename, or comment unrelated code/variables/files.
- Quote paths with spaces (e.g. `"My Drive"`).
- Concise, professional, friendly — but critical when needed.
- Only execute tasks listed in the "Claude" or "Gemini" section of TODO.md unless explicitly asked.
- Use your jupyter notebook tool when editing notebooks.

## Data Integrity
- **Never** modify `Data/Raw/`.
- Never overwrite processed datasets without (1) explicit confirmation and (2) a versioned backup.
- New derived datasets: save in the matching-filetype folder under `Data/Processed/`, name in `snake_case`, and log in `data_description.md` with: description, source file(s), transformation, date.

## Before Running Any File-Modifying Script
1. Describe in plain English what it will do.
2. List all files created, modified, or deleted, with exact output paths.
3. Flag non-reversible actions; confirm if data loss is possible.

## After Running Code
- Summarize what changed (diff-style for key modifications).
- Highlight changed parameters, thresholds, filters, or assumptions.

## Reproducibility
- Fixed random seeds; deterministic where possible.
- Input/output paths declared at top of file; no hardcoded absolute paths unless instructed.
- Log key parameters. Scripts must be idempotent (re-running is safe; `00→01→01→02` = `00→01→02`).
- New dependencies: install in `.venv`, add to `requirements.txt`.

## Destructive Action Safeguards
Never: force-push, `git reset --hard`, `rm -rf`, delete branches, or rewrite commit history. Confirm before any delete/overwrite/force-push.

## Code Changes
Update `code_description.md` when modifying existing code.
Update `pretty_dict.json` when new variables are created. Prompt me to write the pretty variable name if unclear.