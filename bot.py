import rlgym
import time
import numpy as np
from DQNAgent import DQNAgent
import random
import os
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward, BallYCoordinateReward, EventReward
from OurObsBuilder import OurObsBuilder
import controller_states as cs
import action_sets

env = rlgym.make(game_speed=100, obs_builder=OurObsBuilder(), terminal_conditions=[GoalScoredCondition(), TimeoutCondition(1000)], reward_fn=EventReward(goal=1000, concede=-1000, touch=200, shot=700, save=300))
state_size = OurObsBuilder.STATE_SIZE
action_size = cs.CONTROL_STATES_COUNT

batch_size = 32
episode_size = 100
# output_dir = 'model_output/test'
# if not os.path.exists(output_dir):
#   os.makedirs(output_dir)
agent = DQNAgent(state_size, action_size)

for e in range(episode_size):
  state = env.reset()
  tick = 0
  episode_done = False
  if not os.path.exists("save/episode{}/".format(e)):
    os.makedirs("save/episode{}/".format(e))
  f = open('save/episode{}/save.csv'.format(e),'a')
  csv_header = 'TICK,'
  for word in OurObsBuilder.STATE_TITLES:
    csv_header += (word + ',')
    
  csv_header += 'ACTION INDEX,REWARD'
  np.savetxt(f, [csv_header], fmt=''.join(['%s']), delimiter=',')
  while not episode_done:
    action_index = agent.act(state)
    action = action_sets.get_action_set_from_controller_state(cs.controller_states[action_index], state)
    
    next_state, reward, episode_done, _ = env.step(action)

    agent.remember(state, action_index, reward, next_state, episode_done)
    if (tick % 200 == 0):
      csv_print_text = []
      csv_print_text.append(tick)
      csv_print_text.extend(state)
      csv_print_text.extend([action_index, reward])
    
      np.savetxt(f, [csv_print_text], fmt=''.join(['%s']), delimiter=',')


    state = next_state
  
    if episode_done:
        print("Episode {} complete.\nEpsilon: {}".format(e, agent.epsilon))
    tick += 1
  f.close()
  if len(agent.memory) > batch_size:
    agent.replay(batch_size)


  # if e % 2 == 0:
  #   agent.save(output_dir + "weights" + '{:04d}'.format(e) + ".hdf5")