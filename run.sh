python3 -m heareval.tasks.runner speech_commands
python3 -m heareval.tasks.runner nsynth_pitch
python3 -m heareval.tasks.runner dcase2016_task2
python3 heareval/embeddings/runner.py hearbaseline
python3 heareval/predictions/runner.py hearbaseline
python3 heareval/evaluation/runner.py 
