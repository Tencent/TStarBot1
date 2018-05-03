# SC2Learner: StarCraft-II Reinforcement Learning Environment

## Overview

<p align="center">
<img src="docs/images/overview.png" width=800>
<br/>Overview of SC2Learner.
</p>

## Installation

Python 3.5 and SC2-v3.16.1 required.

```bash
pip3 install -r requirements.txt
```

## Quick Start

- Run a random agent, e.g.
```bash
python3 -u eval.py --agent random --difficulty '1'
```

- Train a dqn agent, e.g.
```bash
CUDA_VISIBLE_DEVICES=0 python3 -u train.py --difficulty '2'
```

- Evaluate a dqn agent with multi-processes, e.g.
```bash
CUDA_VISIBLE_DEVICES=0 python3 -u eval_mp.py \
--agent dqn \
--difficulty '2' \
--init_model_path checkpoints/REPLACED_WITH_YOUR_MODEL_NAME
```

- For more help, please:
```bash
python3 eval.py --help
python3 eval_mp.py --help
python3 train.py --help
```
