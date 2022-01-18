import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size

    self.memory = deque(maxlen=2000)

    #discount factor of future rewards
    self.gamma = 0.95
    #exploration rate of agent
    self.epsilon = 1.0
    self.epsilon_decay = 0.995
    self.epsilon_min = 0.01

    self.learning_rate = 0.001

    self.model = self._build_model()

  def _build_model(self):

    model = Sequential()

    model.add(Dense(20, input_dim = self.state_size, activation='tanh'))
    model.add(Dense(15, activation='tanh'))
    model.add(Dense(self.action_size, activation='tanh'))
    
    model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

  def remember(self, state, action, reward, next_state, done):
    self.memory.append(state, action, reward, next_state, done)

  def act(self, state):
    #exploration is chosen
    if np.random.rand() <= self.epsilon:
      action_list = []
      #throttle and steer (pitch and yaw)
      for i in range(0, 2):
        action_list.append(np.random.random.random_integers(-1, 1))
      #jump, boost, brake
      for i in range(0, 3):
        action_list.append(np.random.random.random_integers(-1, 1))
      
      return action_list

    #exploitation is chosen
    else:
      action_list = self.model.predict(state)
      return action_list

  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
      target = reward
      if not done:
        target = reward + self.gamma * self.model.predict(next_state)
      target_f = self.model.predict(state)
      target_f[0][action] = target

      self.model.fit(state, target_f, epochs=1, verbose=0)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)

  def interperet_action_list(self, action_list, state):
    final_action_list = np.ndarray((1, 8))

    car_on_ground = state.is_on_ground
    car_in_air = not car_on_ground
    #Action List
    #throttle, steering, jump, roll, brake
    
    #RLGym action parser
    #throttle, steer, yaw, pitch, roll, jump, boost, handbrake
    if car_on_ground:
      #copy over values for throttle and steering
      for i in range(0, 2):
        np.put(final_action_list, i, action_list[i])
      #yaw, pitch, and roll cannot be adjusted when car is on ground
      for i in range (2, 5):
        np.put(final_action_list, i, 0)
      #copy over values for jump, boost, and handbrake
      for i in range (5, 8):
        np.put(final_action_list, i, action_list[i - 3])

    elif car_in_air:
      #throttle and steer are 0
      for i in range(0, 2):
        np.put(final_action_list, i, 0)
      #roll is activated, use steering value to control roll
      #yaw cannot be altered when rolling
      #copy throttle to control pitch
      #copy steering to control roll
      if action_list[4] == 1:
        np.put(final_action_list, 2, 0)
        np.put(final_action_list, 3, action_list[0])
        np.put(final_action_list, 4, action_list[1])
      #roll is NOT activated
      #copy steering to control yaw
      #copy throttle to control pitch
      #roll cannot be altered when NOT activated
      else:
        np.put(final_action_list, 2, action_list[1])
        np.put(final_action_list, 3, action_list[0])
        np.put(final_action_list, 4, 0)
      #jump and boost match jump and boost values passed in
      for i in range(5, 7):
        np.put(final_action_list, i, action_list[i - 3])
      #handbrake cannot be used while car is not on ground
      np.put(final_action_list, 7, 0)