import rlgym
import time
import numpy as np
from DQNAgent import DQNAgent
import random
import os
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward, BallYCoordinateReward, EventReward
from OurObsBuilder import OurObsBuilder
import control_states as cs

print("hello...")

env = rlgym.make(obs_builder=OurObsBuilder(), terminal_conditions=[GoalScoredCondition(), TimeoutCondition(1000)], reward_fn=EventReward(goal=1000, concede=-1000, touch=200, shot=700, save=300))
print("make command should be executed now")
state_size = 16
action_size = 72

batch_size = 32
episode_size = 100
# output_dir = 'model_output/test'
# if not os.path.exists(output_dir):
#   os.mkdirs(output_dir)
agent = DQNAgent(state_size, action_size)

for e in range(episode_size):
  state = env.reset()
  
  episode_done = False
  while not episode_done:
    action_index = agent.act(state)
    action = cs.control_states[action_index]
    
    next_state, reward, episode_done, _ = env.step(action)

    agent.remember(state, action, reward, next_state, episode_done)

    state = next_state
  
    if episode_done:
        print("Episode {} complete.\nEpsilon: {}".format(e, agent.epsilon))
  if len(agent.memory) > batch_size:
    agent.replay(batch_size)


#   if e % 50 == 0:
#     agent.save(output_dir + "weights" + '{:04d}'.format(e) + ".hdf5")