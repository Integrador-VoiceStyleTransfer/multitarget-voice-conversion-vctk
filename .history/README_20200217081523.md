# Voice conversion via disentangle context representation (Code for paper:https://arxiv.org/pdf/1804.02812.pdf )

# Steps for training:
1. Use make_dataset_vctk.py to convert the VCTK corpus into training data.
2. Use make_single_samples.py to make the data feed dictionary/log used in training.
3. Train using main.py
4. Generate converted samples using convert.py

## Audio convertion:
We have taken the original ``test.py`` from Chou's implementation an added a few options.

To perform a convertion you must execute the command

``python test.py -m {model_path} [args]``

The passed ``args`` define the conversion mode.

| Option | Source | Target |
|---|---|---|
| One vs. One | ``-s`` | ``-t`` |
| One vs. All | ``-s`` | ``-tl`` |
| All vs. One | ``-sl`` | ``-t`` |
| One vs. All | ``-sl`` | ``-tl`` |

Where
- ``-s`` source audio file.
- ``-sl`` source list file. Each line must be the path of the source audio file.
- ``-t`` target speaker id
- ``-tl`` target list file. Each line must be the id of the desired target speaker.