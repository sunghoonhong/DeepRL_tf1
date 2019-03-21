# Snake (Discrete action space)

![PPO agent after 50000 episodes](https://github.com/sunghoonhong/DeepRL/blob/master/snake_feature/gifs/ppo%20after%2050000%20episode.gif)

## Feature Extraction
### State(size = 8)
- Direction vector (2): represents 4 direction (up, down, left, right)
- Relative vector between head and goal (2): head - goal vector
- Safe distance towards 3 possible direction (3): straight, left, right
- Length of snake(1): starts from 1

### Action(size = 3)
- 0: Go straight
- 1: Turn left
- 2: Turn right

### Reward Scheme
- Get goal: +1
- Die(boundary, body): -1
- Get closer: +0.01
- Get farther: -0.01


## Issue
1. Too dependent reward scheme  
Giving rewards when snake gets closer or farther seems to be too dependent for this environment.  
But, It makes learning much faster.  
Giving +1 for goal, otherwise -1 is too sparse for learning in short time.  
I've been trying to get rid of distance from reward scheme.  

2. Only 3 action  
Human usually plays 'snake' with 4 arrow keys.  
So, there are 4 actions including key which represents reverse direction makes no difference.  
(Actually, there are only 3 actions that affect snake's move.)  
Although action space of which size is 3 is more easier, I want to make action space of which size is 4, and then figure out how is gonna policy for reverse action.  