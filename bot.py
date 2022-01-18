import rlgym
import time
import numpy as np
import DQNAgent
import OurObsBuilder
import random
import os

from OurObsBuilder import OurObsBuilder

print("hello...")

env = rlgym.make(obs_builder=OurObsBuilder())

while True:
    obs = env.reset()
    print(obs)
    done = False
    steps = 0
    ep_reward = 0
    t0 = time.time()
    while not done:
        actions = env.action_space.sample()  # agent.act(obs) | Your agent should go here
        new_obs, reward, done, state = env.step(actions)
        ep_reward += reward
        obs = new_obs
        steps += 1

    length = time.time() - t0
    #print("Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(length / steps, length, ep_reward))

exit()
# env = rlgym.make()
# print("make command should be executed now")
# state_size = 12
# action_size = 72

# batch_size = 32
# episode_size = 100
# output_dir = 'model_output/test'
# if not os.path.exists(output_dir):
#   os.mkdirs(output_dir)
# agent = DQNAgent(state_size, action_size)

# for e in range(episode_size):
#   state = env.reset()
#   print(state[0])
#   state = np.reshape(state, [1, state_size])
#   print(state)
#   episode_done = False
  # while not episode_done:
  #   action = agent.act(state)
  
  #   next_state, reward, done, _ = env.step(action)

  #   next_state = np.reshape(next_state, [1, next_state])

  #   agent.remember(state, action, reward, next_state, done)

  #   state = next_state
  
  # if len(agent.memory) > batch_size:
  #   agent.replay(batch_size)

  # if e % 50 == 0:
  #   agent.save(output_dir + "weights" + '{:04d}'.format(e) + ".hdf5")