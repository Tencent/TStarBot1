# SC2LAB: StarCraftII Lab.

## Installation
```bash
pip install -r requirements.txt
```
## Examples

- Run Test:
```bash
python -m tests.test_sc2_env
```

- CartPoleV0 Example:
```bash
CUDA_VISIBLE_DEVICES= python -u run_cartpole_dqn.py --agent dqn
```

- StarCraftII Example:
```bash
CUDA_VISIBLE_DEVICES=0 python -u run_sc2_terran_dqn.py --agent dqn
```
