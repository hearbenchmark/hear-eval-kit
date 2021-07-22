# hear2021-eval-kit

Evaluation kit for HEAR 2021 NeurIPS competition

See [ROADMAP](ROADMAP.md).

## Usage

```
pip install heareval
```

### Evaluation Tasks
To run the preprocessing pipeline for Google Speech Commands:
```
python3 -m heareval.tasks.runner speech_commands
```

For NSynth pitch:
```
python3 -m heareval.tasks.runner nsynth_pitch
```

These commands will download and preprocess the entire dataset. An intermediary dir
call `_workdir` will be created, and then a final directory called `tasks` will contain
the completed dataset.

Options:
```
Options:
  --num-workers INTEGER  Number of CPU workers to use when running. If not
                         provided all CPUs are used.

  --sample-rate INTEGER  Perform resampling only to this sample rate. By
                         default we resample to 16000, 22050, 44100, 48000.
```



[later we will include more details here]

## Development

Clone repo:
```
git clone https://github.com/neuralaudio/hear2021-eval-kit
cd hear2021-eval-kit
```
Install in development mode:
```
pip install -e ".[dev]"
```

Make sure you have pre-commit hooks installed:
```
pre-commit install
```

Running tests:
```
python3 -m pytest
```
