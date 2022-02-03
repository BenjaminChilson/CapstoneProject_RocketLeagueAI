import rlgym
from datetime import datetime
import numpy as np
from DQNAgent import DQNAgent
import random
import os
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward, BallYCoordinateReward, EventReward
from OurObsBuilder import OurObsBuilder
import controller_states as cs
import action_sets
import bot_helper_functions as bhf

env = rlgym.make(game_speed=100, obs_builder=OurObsBuilder(), terminal_conditions=[GoalScoredCondition()], reward_fn=EventReward(goal=1000, concede=-1000, touch=200, shot=700, save=300))
state_size = OurObsBuilder.STATE_SIZE
action_size = cs.CONTROL_STATES_COUNT

batch_size = 32
episode_size = 100

agent = DQNAgent(state_size, action_size)

run = 1
while run == 1:
  # structure will be /save/timestamp/episode/#
  # every episode_size amount of episodes, a new timestamp is generated so we don't overwrite files
  training_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  agent.save_weight_as_csv("save/{}/episode/{}/".format(training_timestamp, 0), "initial_weights.csv")
  
  # run episodes
  for e in range(episode_size):
    start_time = datetime.now().strftime("%Y:%m:%d_%H:%M:%S")
    if not os.path.exists("save/{}/model/".format(training_timestamp)):
      os.makedirs("save/{}/model/".format(training_timestamp))
    agent.save("save/{}/model/start_of_training_weights.hdf5".format(training_timestamp))

    state = env.reset()
    tick = 0
    total_reward = 0
    episode_done = False
    
    while not episode_done:
      action_index = agent.act(state)
      action = action_sets.get_action_set_from_controller_state(cs.controller_states[action_index], state)
      
      next_state, reward, episode_done, _ = env.step(action)

      agent.remember(state, action_index, reward, next_state, episode_done)

      state = next_state

      total_reward += reward
    
      if episode_done:
        agent.save_weight_as_csv("save/{}/episode/{}/".format(training_timestamp, e), "final_weights.csv")
        
        bhf.save_training_results_as_csv(training_timestamp, e, tick, total_reward, start_time)
        
        print("Episode {} complete.\nEpsilon: {}".format(e, agent.epsilon))
      
      tick += 1

    if len(agent.memory) > batch_size:
      agent.replay(batch_size)

  # save weights file after every episodes_size episodes
  agent.save("save/{}/model/end_of_training_weights.hdf5".format(training_timestamp))