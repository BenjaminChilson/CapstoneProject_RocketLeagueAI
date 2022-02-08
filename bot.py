import rlgym
from datetime import datetime
import numpy as np
from DQNAgent import DQNAgent
import random
import os
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, NoTouchTimeoutCondition
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward, BallYCoordinateReward, EventReward, RewardIfBehindBall
from rlgym.utils.reward_functions.combined_reward import CombinedReward
from OurObsBuilder import OurObsBuilder
from GroupRewardFunction import GroupRewardFunction
import controller_states as cs
import action_sets
import bot_helper_functions as bhf


# tick-skip = 40 = 3 actions per second
rlgym.make(game_speed=100, tick_skip=30, obs_builder=OurObsBuilder(), terminal_conditions=[GoalScoredCondition(), NoTouchTimeoutCondition(max_steps=225)], reward_fn=GroupRewardFunction())
obs_state_size = OurObsBuilder.STATE_SIZE
agent_state_size = obs_state_size - 1
action_size = cs.CONTROL_STATES_COUNT

batch_size = 1000
episode_size = 250

agent = DQNAgent(agent_state_size, action_size)

run = 1
while run == 1:
  # structure will be /save/timestamp/episode/#
  # every episode_size amount of episodes, a new timestamp is generated so we don't overwrite files
  training_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  agent.save_weight_as_csv("save/{}/episode/{}/".format(training_timestamp, 0), "initial_weights.csv")
  if not os.path.exists("save/{}/model/".format(training_timestamp)):
    os.makedirs("save/{}/model/".format(training_timestamp))
  agent.save("save/{}/model/start_of_training_weights.hdf5".format(training_timestamp))

  # run episodes
  for e in range(episode_size):
    start_time = datetime.now().strftime("%Y:%m:%d_%H:%M:%S")
    state = env.reset()
    tick = 0
    total_reward = 0
    episode_done = False
    
    while not episode_done:
      #removes car_on_ground value from state, not required for act() but is required for getting the action set
      car_on_ground = state.pop(OurObsBuilder.CAR_ON_GROUND_INDEX)
      action_index = agent.act(state)
      action = action_sets.get_action_set_from_controller_state(cs.controller_states[action_index], car_on_ground)
      
      next_state, reward, episode_done, _ = env.step(action)

      #this is done, because the memory of the state shouldn't include the on_ground value, similar to the state variable above
      reduced_next_state = next_state.copy()
      reduced_next_state.pop(OurObsBuilder.CAR_ON_GROUND_INDEX)

      agent.remember(state, action_index, reward, reduced_next_state, episode_done)

      state = next_state

      total_reward += reward
      
      if episode_done:
        print("episode ended at tick {}".format((tick + 1)/15))
        bhf.save_training_results_as_csv(training_timestamp, e, tick, total_reward, start_time)
        print("Episode {} complete.\nEpsilon: {}".format(e, agent.epsilon))

      tick += 1
    
    if len(agent.memory) > batch_size:
      agent.replay(batch_size)
    agent.save_weight_as_csv("save/{}/episode/{}/".format(training_timestamp, e), "final_weights.csv")

  # save weights file after every episodes_size episodes
  agent.save("save/{}/model/end_of_training_weights.hdf5".format(training_timestamp))

  agent.reset_epsilon()
