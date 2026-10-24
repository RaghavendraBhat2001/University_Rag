# uniRAG speed + fixes: what to do

Put EVERY file from this zip in your project folder (the one with app.py), e.g.
C:\Users\ragha\Desktop\University_Rag\   Then open PowerShell there and activate your venv.

## Step 0 - before anything
- Close the app. Keep Ollama running. Plug in the charger, set Windows power mode to "Best performance".
- Update Ollama from ollama.com (installer), then restart it.
- Back up the project folder (copy it) in case you want to undo.

## Step 1 - find the best CPU thread count (5 min)
    python benchmark_threads.py
If it says "In config.py set: LLM_NUM_THREAD = N", edit config.py: change `LLM_NUM_THREAD = None` to that number.
If it says to leave it, do nothing.

## Step 2 - apply the patches (each makes its own backup, each refuses to run twice)
    python patch_parse_cache.py      # repeated questions skip the 30 s understanding step
    python patch_lab_subject.py      # lab subjects (DS LAB etc.) give code for exam-style questions
    python patch_cutoff.py           # related-questions cutoff 1.15 -> 0.75 (measured, see calibration)
If any says "does not look the way this patch expects", stop and send me that file.

## Step 3 - check quality did not drop (takes a while: fresh answers)
1. In config.py set `ANSWER_CACHE = False`
2. python eval_questions.py
3. Compare the PASS/FAIL/WARN with your earlier runs. Then set `ANSWER_CACHE = True` again.

## Step 4 - measure the cutoff (1 min, optional but good for the report)
    python calibrate_cutoff.py --current 0.75
Keep the printed table for your report.

## Step 5 - tidy the folder (do not delete yet, MOVE)
    mkdir archive
    Move-Item *.pre_* archive\
    Move-Item *.bak* archive\
    Move-Item students.json.corrupt-* archive\
    Move-Item students.json.mine archive\
    Move-Item patch_*.py archive\          # AFTER step 2 is finished
    Move-Item tidy_*.py archive\
    Move-Item apply_verified.py, add_recovered_questions.py archive\
Start the app once and ask one question to confirm it still works.
Also open students.json and remove your own account before handing the project over.

## Step 6 - overnight: pre-compute answers
Edit warm_questions.txt (add the questions you expect), then:
    python warm_cache.py
Leave the laptop plugged in and do not use the app meanwhile. Re-run any time: finished questions return from the cache in seconds.

## Expected result
- New questions: still about 2-4 minutes (CPU limit).
- Repeated or pre-warmed questions: seconds.
- Lab questions in DS LAB etc. return code.
- Answer quality: unchanged (same model, same prompts, same answer lengths).

## Files in this zip
benchmark_threads.py, patch_parse_cache.py, patch_lab_subject.py, patch_cutoff.py,
calibrate_cutoff.py + calibration_set.csv, warm_cache.py + warm_questions.txt
