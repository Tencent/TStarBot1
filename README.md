# SC2LAB: StarCraftII Lab.

## Installation
```bash
pip install -r requirements.txt
```
## Examples

- Test Envs, e.g.
```bash
python -m tests.test_sc2_env
```

- Train CartPoleV0, e.g.
```bash
CUDA_VISIBLE_DEVICES= python -u train_cartpole_dqn.py --agent dqn
```

- Train StarCraftII, e.g.
```bash
CUDA_VISIBLE_DEVICES=0 python -u train_sc2_terran_dqn.py --agent dqn
```

- Evaluate StarCraftII, e.g.
```bash
CUDA_VISIBLE_DEVICES=0 python -u evaluate_sc2_terran.py --agent random --difficulty '2'
CUDA_VISIBLE_DEVICES=0 python -u evaluate_sc2_terran.py --agent dqn --difficulty '2' --init_model_path checkpoints/REPLACED_WITH_YOUR_MODEL_NAME
```
