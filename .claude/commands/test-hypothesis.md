Run the full hypothesis testing pipeline for the given hypothesis ID (e.g. H1, H2).

## Steps — execute in order, do not skip any

### 1. Read the hypothesis
Read `HYPOTHESES.md` and find the entry matching the given ID. Extract:
- The exact code change required
- The expected effect

### 2. Log to experiment tracker
Invoke the `ml-tracker` agent to log the experiment to `EXPERIMENT_LOG.md` before any code changes. Pass the hypothesis description, expected effect, and current hyperparameters from `train.py`.

### 3. Mark as in progress
In `HYPOTHESES.md`, change the hypothesis status from `[ ]` to `[~]`.

### 4. Implement the code change
Make the minimum change described in the hypothesis. Read the relevant file(s) first.

### 5. Run training in the background
```bash
source .venv/bin/activate && python train.py 2>&1 | tee logs/hypothesis_run.log
```
Run this with `run_in_background=True`. Do not wait — continue only when notified of completion.

### 6. Parse the result
Read `logs/hypothesis_run.log` to extract the final val Sharpe.

### 7. Evaluate and update records
- Update `HYPOTHESES.md`: set status to `[x]`, fill in Result and Conclusion
- Invoke `ml-tracker` to close the experiment in `EXPERIMENT_LOG.md` with the result and conclusion
- If the hypothesis is **disproven** (val Sharpe lower than baseline): revert the code change

### 8. Send notification
```bash
osascript -e 'display notification "HYPOTHESIS_SUMMARY" with title "RL Agent" sound name "Glass"'
```
Replace `HYPOTHESIS_SUMMARY` with: `"H{ID} complete — val Sharpe {result} (baseline {baseline}). {Confirmed/Disproven}."`
