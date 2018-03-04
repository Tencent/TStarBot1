# SC2LAB: StarCraftII Lab

## Installation

Python 3.6 required.

```bash
pip3 install -r requirements.txt
```
## Examples

- Test Envs, e.g.
```bash
python3 -m tests.test_sc2_env
```

- Train CartPoleV0, e.g.
```bash
CUDA_VISIBLE_DEVICES= python -u train_cartpole_dqn.py --agent dqn
```

- Train StarCraftII, e.g.
```bash
CUDA_VISIBLE_DEVICES=0 python3 -u train_sc2_terran_dqn.py --agent dqn
```

- Evaluate StarCraftII, e.g.
```bash
CUDA_VISIBLE_DEVICES= python3 -u evaluate_sc2_terran.py --agent random --difficulty '2'
CUDA_VISIBLE_DEVICES=0 python3 -u evaluate_sc2_terran.py --agent dqn --difficulty '2' --init_model_path checkpoints/REPLACED_WITH_YOUR_MODEL_NAME
```

- Play StarCraftII with Keyboard, e.g.
```bash
CUDA_VISIBLE_DEVICES= python3 -u evaluate_sc2_terran.py --agent keyboard --difficulty '1'
```
