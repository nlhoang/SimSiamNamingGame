# SimSiam Naming Game

Code for two papers built around SSNG, a self-supervised signaling-game model where two (or more) agents learn a discrete "language" to communicate about their inputs while training a SimSiam-style self-supervised objective.

| Paper | Reference | Entry point | Model |
|---|---|---|---|
| Original (2-agent) | *SimSiam Naming Game: A Unified Approach for Emergent Communication and Representation Learning* ([arXiv:2410.21803](https://arxiv.org/abs/2410.21803)) | `main_ssng.py` | `model/SSNG.py` |
| Generalized (N-agent) | *Generalized SimSiam Naming Game for Multi-Agent Emergent Communication* (IEEE ICDL 2026) | `main_gssng.py` | `model/GSSNG.py` (`MultiAgentSSNG`) |

## Repository layout

```
base/         shared building blocks: encoders/backbones (base_model.py, cifar_resnet.py),
              data loading (dataloader.py), training/eval loop + feature extraction (evaluation.py),
              discrete communication channel (langCoder.py) shared by all SSNG variants, misc helpers (utils.py)
config.py     dataset registry, dataloader factories, per-dataset hyperparameters, seeding
model/        model definitions (see table below)
main_ssng.py  original paper — train/evaluate the 2-agent emergent-communication SSNG model
main_gssng.py generalized paper — train/evaluate the N-agent SSNG model
visualizations/       example emergent-language outputs (signs) for a set of sample inputs
```

## Models

| File             | Class             | Used by | Role |
|------------------|-------------------|---|---|
| `model/SSNG.py`  | `SSNG`            | `main_ssng.py` | 2-agent emergent-communication model (the original paper's main contribution) |
| `model/GSSNG.py` | `MultiAgentSSNG` | `main_gssng.py` | N-agent extension (the generalized paper's main contribution) |

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for practical training times (`torch.amp.autocast`/`GradScaler` assume `cuda`); falls back to CPU otherwise.

Before running anything, update the dataset folder path in `config.py`:

```python
path = '/home/nlhoang/MachineLearning/data/'  # <- change to your local dataset directory
```

This is where `MNIST`/`FashionMNIST`/`KMNIST`/`CIFAR`/`CIFAR100` are expected to live (downloaded automatically on first run if missing). `ImageNet100` is instead pulled from the Hugging Face Hub (`clane9/imagenet-100`) and does not use this path.

For `ImageNet100`, also update the class subset used for training/evaluation in `config.py`:

```python
ImgNet_select = list(range(1, 21))
```

This is a hardcoded list of 20 class indices (out of the 100 in `clane9/imagenet-100`) used whenever `--dataset ImageNet100` is selected. Replace it with your own list of class indices.

## Usage

Each entry point writes logs and checkpoints to a fresh timestamped directory under `experiments/` (created via `mkdtemp`), and prints its device, model size, and training/eval progress to that directory's `run.log`.

### Original paper — emergent communication (`main_ssng.py`)

Trains the 2-agent `SSNG` model with a discrete communication channel, then evaluates classification accuracy from the learned signs.

```bash
python main_ssng.py --dataset CIFAR --EmCom EmCom --eval-epochs 50
```

Key flags: `--dataset` (`MNIST`, `FashionMNIST`, `KMNIST`, `CIFAR`, `CIFAR100`, `ImageNet100`), `--EmCom` (`EmCom` to train with communication, `NoCom` for the no-communication ablation), `--seed`, `--interval-saved`, `--debug` (2-epoch smoke test).

### Generalized paper — N-agent SSNG (`main_gssng.py`)

Trains `MultiAgentSSNG` with a configurable number of communicating agents.

```bash
python main_gssng.py --num-of-agent 4 --dataset MNIST --EmCom EmCom
```

Key flags: `--num-of-agent` (>= 2, default 3), `--dataset` (`MNIST`, `FashionMNIST`, `KMNIST`, `CIFAR`, `CIFAR100`, `ImageNet100`), `--EmCom` (`EmCom`/`NoCom`), `--debug` (2-epoch smoke test, default `False`).

## Citation

If you use this code, please cite the corresponding paper:

```bibtex
@article{hoang2024simsiamnaminggame,
  title   = {SimSiam Naming Game: A Unified Approach for Emergent Communication and Representation Learning},
  author  = {Nguyen Le Hoang and Tadahiro Taniguchi and Tianwei Fang and Akira Taniguchi and Masatoshi Nagano},
  journal = {arXiv preprint arXiv:2410.21803},
  year    = {2024}
}

@inproceedings{hoang2026generalizedssng,
  title     = {Generalized SimSiam Naming Game for Multi-Agent Emergent Communication},
  author    = {Nguyen Le Hoang and Tadahiro Taniguchi and Tianwei Fang and Masatoshi Nagano},
  booktitle = {IEEE International Conference on Development and Learning (ICDL)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
