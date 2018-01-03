# Primitive Experimental Results for StarCraft-II

## Mini-Games with Simplified Obs/Acation Spaces

<center>

MiniGame                | Ours-Simplified   |  Random   | DeepMind | Human
:---------------------  | ---------------:  | -----:    | ----:    | -----:
*MoveToBeacon*          |     26.1          |   1       | 26       | 26
*CollectMineralShards*  |     91.9          |   17      | 103      | 133 
*DefeatRoaches*         |     81.6          |   1       | 100      | 41

</center>

Possible reasons for the current performance gap:

- **Not Fully Converged**: 60M game steps in our results VS. 600M in DeepMind's.
- **No Hyper-paramter Tuning**: DeepMind conducted a careful random hyper-parameter selection, while in our's, no hyper-parameter selection was performed.
- **Simplified Observation and action spaces**: we only use a single feature layer (*screen.relative\_player*) and a single action (*move\_attack*), while DeepMind utilizes the full spaces.

<img src="images/MoveToBeacon.gif" width=270> <img src="images/CollectMineralShards.gif" width=270> <img src="images/DefeatRoaches.gif" width=270>

## Mini-Games with Full Obs/Acation Spaces

(On-going)

MiniGame                | Ours-Simplified   |  Ours-Full  | Random   | DeepMind | Human
:---------------------  | ---------------:  | ----------: | -----:   | ----:    | -----:
*MoveToBeacon*          |     26.1          |   -         |  1       | 26       | 26
*CollectMineralShards*  |     91.9          |   -         |  17      | 103      | 133 
*DefeatRoaches*         |     81.6          |   -         |  1       | 100      | 41
