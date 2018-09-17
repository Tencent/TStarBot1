# SC2Learner (TStarBot1) - *Macro-action*-based StarCraft-II Reinforcement Learning Environment


<p align="center">
<img src="docs/images/overview.png" width=750>
</p>

*[SC2Learner](https://github.com/Tencent-Game-AI/sc2learner)* is a *macro-action*-based [StarCraft-II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty) reinforcement learning platform for research.
It exposes the re-designed StarCraft-II action space, which has more than a hundred  descrete macro actions, based on the raw APIs exposed by DeepMind and Blizzard's [PySC2](https://github.com/deepmind/pysc2).
The redesign of macro action space relieves the learning algorithms from a disastrous burden of directly handling a massive number of atomic keyboard and mouse operations, making learning more tractable.
The environments and wrappers strictly follow the inferface of [OpenAI Gym](https://github.com/openai/gym), making it easier to be adapted to many off-the-shelf reinforcement learning algorithms and implementations.

[*TStartBot1*](https://whitepaper-url-to-be-added), a reinforcement learning agent, is also released, with two off-the-shelf reinforcement learning algorithms *Dueling Double Deep Q Network* (DDQN) and *Proximal Policy Optimization* (PPO), as examples. **Distributed** versions of both algorithms are released, enabling learners to scale up the rollout precedures across thousands of CPU cores in a cluster of machines. We also release a well-trained PPO model, which is able to beat **level-9** built-in AI (cheating resources) with **97%** win-rate and **level-10** (cheating insane) with **81%** win-rate.  

A whitepaper is available now at [here](https://whitepaper-url-to-be-added), and here's a BibTeX entry that you can use to cite it in a publication: TO BE ADDED.

## Dependencies

- Python >= 3.5.2 required.

- PySC2-folk: https://github.com/Tencent-Game-AI/pysc2.git

## Installation

```bash
pip3 install -e .
```

## How to Run

### Run Random Agent
```bash
python3 -m sc2learner.bin.evaluate --agent random --difficulty '1'
```

### Train PPO Agent

- Start Actors
```bash
for i in $(seq 0 128); do
  python3 -m sc2learner.bin.train_ppo --job_name=actor --learner_ip localhost &
done;
```

- Start Learner
```bash
python3 -m sc2learner.bin.train_ppo --job_name learner
```

### Evaluate PPO Agent
```bash
python3 -m sc2learner.bin.eval --agent ppo --model_path REPLACE_WITH_YOUR_OWN_MODLE_PATH
```
###

### Play vs. PPO Agent

- Start Human Player's Client
```bash
python3 -m pysc2.bin.play_vs_agent --human --map AbyssalReef --user_race zerg
```

- Start PPO Agent
```bash
python3 -m sc2learner.bin.play_vs_ppo_agent --model_path REPLACE_WITH_YOUR_OWN_MODLE_PATH
```

### Selfplay Train PPO Agent

- Start Actors
```bash
for i in $(seq 0 128); do
  python3 -m sc2learner.bin.train_ppo_selfplay --job_name=actor --learner_ip localhost &
done;
```

- Start Learner
```bash
python3 -m sc2learner.bin.train_ppo_selfplay --job_name learner
```
