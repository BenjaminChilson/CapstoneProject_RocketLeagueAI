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

env = rlgym.make(game_speed=100, obs_builder=OurObsBuilder(), terminal_conditions=[GoalScoredCondition()], reward_fn=EventReward(goal=1000, concede=-1000, touch=200, shot=700, save=300))
state_size = OurObsBuilder.STATE_SIZE
action_size = cs.CONTROL_STATES_COUNT

batch_size = 32
episode_size = 100

# output_dir = 'model_output/test'
# if not os.path.exists(output_dir):
#   os.makedirs(output_dir)
agent = DQNAgent(state_size, action_size)

run = 1
while run == 1:
  # structure will be /save/timestamp/episode/#
  # every episode_size amount of episodes, a new timestamp is generated so we don't overwrite files
  timestamp = int(time.time() * 1000.0)

  # run episodes
  for e in range(episode_size):
    state = env.reset()
    tick = 0
    episode_done = False
    if not os.path.exists("save/{}/episode{}/".format(timestamp, e + 1)):
      os.makedirs("save/{}/episode{}/".format(timestamp, e + 1))
    f = open('save/{}/episode{}/save.csv'.format(timestamp, e + 1),'a')
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
          print("Episode {} complete.\nEpsilon: {}".format(e + 1, agent.epsilon))
      tick += 1
    f.close()
    if len(agent.memory) > batch_size:
      agent.replay(batch_size)

  # save weights file after every episodes_size episodes
  agent.save("save/{}/model/weights.hdf5".format(timestamp))