import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
# import ActionSets

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

    model.add(Dense(20, input_dim = self.state_size, activation='relu', name="first_hidden_layer"))
    model.add(Dense(15, activation='relu', name="second_hidden_layer"))
    model.add(Dense(self.action_size, activation='relu', name="output_layer"))
    
    model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))



    return model

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def act(self, state):
    #exploration is chosen
    if np.random.rand() <= self.epsilon:
      # print("Exploration is Chosen")
      action_list = []
      #throttle and steer (pitch and yaw)
      for i in range(0, 2):
        action_list.append(np.random.random_integers(-1, 1))
      #jump, boost, brake
      for i in range(0, 3):
        action_list.append(np.random.random_integers(0, 1))
      interpretpted_action_set = self.interperet_shortened_action_list(action_list, state)
      # print("ias: {}".format(interpretpted_action_set))
      return interpretpted_action_set

    #exploitation is chosen
    else:
      updated_state = np.reshape(state, [1, self.state_size])
      predictions = self.model.predict(updated_state)[0]
      # print(len(predictions))
      # print(predictions)
      # print(np.argmax(predictions))
      action_set_index = np.argmax(predictions)
      action_set = self.possible_action_sets[action_set_index]
      interpretpted_action_set = self.interperet_shortened_action_list(action_set, state)
      return interpretpted_action_set

  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    for (state, action, reward, next_state, done) in minibatch:
      target = reward

      next_state = np.reshape(next_state, [1, self.state_size])
      state = np.reshape(state, [1, self.state_size])


      if not done:
        target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

      target_f = self.model.predict(state)
      action = self.decode_action_set(action)
      action_index = self.possible_action_sets.index(action)
      target_f[0][action_index] = target

      self.model.fit(state, target_f, epochs=1, verbose=0)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)

  def interperet_shortened_action_list(self, shortened_action_set, state):
    # print("given action list: {}".format(action_list))
    interpreted_action_set = []

    car_on_ground = state[15]
    car_in_air = not car_on_ground
    #Action List
    #throttle, steering, jump, roll, brake
    
    if car_on_ground:
      #copy over values for throttle and steering
      for i in range(0, 2):
        interpreted_action_set.append(shortened_action_set[i])
      #yaw, pitch, and roll cannot be adjusted when car is on ground
      for i in range (2, 5):
        interpreted_action_set.append(0)
      #copy over values for jump, boost, and handbrake
      for i in range (2, 5):
        interpreted_action_set.append(shortened_action_set[i])
      
      return interpreted_action_set

    elif car_in_air:
      roll_activated = shortened_action_set[4] == 1
      roll_not_activated = not roll_activated

      #throttle and steer are 0
      for i in range(0, 2):
        interpreted_action_set.append(0)
      #roll is activated, use steering value to control roll
      #yaw cannot be altered when rolling
      #copy throttle to control pitch
      #copy steering to control roll
      if roll_activated:
        interpreted_action_set.append(0)
        interpreted_action_set.append(shortened_action_set[0])
        interpreted_action_set.append(shortened_action_set[1])
      #roll is NOT activated
      #copy steering to control yaw
      #copy throttle to control pitch
      #roll cannot be altered when NOT activated
      elif roll_not_activated:
        interpreted_action_set.append(shortened_action_set[1])
        interpreted_action_set.append(shortened_action_set[0])
        interpreted_action_set.append(0)
      #jump and boost match jump and boost values passed in
      for i in range(2, 4):
        interpreted_action_set.append(shortened_action_set[i])
      #handbrake cannot be used while car is not on ground
      interpreted_action_set.append(0)

      return interpreted_action_set

  #throttle, steer, yaw, pitch, roll, jump, boost, handbrake
  def decode_action_set(self, action_set):
    new_action_set = []
    decoded_action_set = []
    action_set = np.reshape(action_set, (1, 8))
    
    for i in range(0, 8):
      new_action_set.append(action_set[0][i])
    
    #action_set[0] will be controlling either throttle or pitch
    for x in (new_action_set[0], new_action_set[3]):
      if x != 0:
        decoded_action_set.append(x)
        break
    if new_action_set[0] == 0 and new_action_set[3] == 0:
      decoded_action_set.append(0)
    #action_set[1] will be controlling either steering, yaw, or rolling
    for x in (new_action_set[1], new_action_set[2], new_action_set[4]):
      if x != 0:
        decoded_action_set.append(x)
        break
    if new_action_set[1] == 0 and new_action_set[2] == 0 and new_action_set[4] == 0:
      decoded_action_set.append(0)

    decoded_action_set.append(new_action_set[5])
    decoded_action_set.append(new_action_set[6])
    decoded_action_set.append(new_action_set[7])

    return decoded_action_set
    


#{Throttle, Steering, Jump, Boost, Brake/Powerslide}
#{Pitch,    Yaw,      Jump, Boost, Brake(Roll?)}

  possible_action_sets = [[0, 0, 0, 0, 0],
  [0, 0, 0, 0, 1],
  [0, 0, 0, 1, 0],
  [0, 0, 0, 1, 1],
  [0, 0, 1, 0, 0],
  [0, 0, 1, 0, 1],
  [0, 0, 1, 1, 0],
  [0, 0, 1, 1, 1],

  [0, 1, 0, 0, 0],
  [0, 1, 0, 0, 1],
  [0, 1, 0, 1, 0],
  [0, 1, 0, 1, 1],
  [0, 1, 1, 0, 0],
  [0, 1, 1, 0, 1],
  [0, 1, 1, 1, 0],
  [0, 1, 1, 1, 1],

  [1, 0, 0, 0, 0],
  [1, 0, 0, 0, 1],
  [1, 0, 0, 1, 0],
  [1, 0, 0, 1, 1],
  [1, 0, 1, 0, 0],
  [1, 0, 1, 0, 1],
  [1, 0, 1, 1, 0],
  [1, 0, 1, 1, 1],

  [1, 1, 0, 0, 0],
  [1, 1, 0, 0, 1],
  [1, 1, 0, 1, 0],
  [1, 1, 0, 1, 1],
  [1, 1, 1, 0, 0],
  [1, 1, 1, 0, 1],
  [1, 1, 1, 1, 0],
  [1, 1, 1, 1, 1],

  [0, -1, 0, 0, 0],
  [0, -1, 0, 0, 1],
  [0, -1, 0, 1, 0],
  [0, -1, 0, 1, 1],
  [0, -1, 1, 0, 0],
  [0, -1, 1, 0, 1],
  [0, -1, 1, 1, 0],
  [0, -1, 1, 1, 1],

  [-1, 0, 0, 0, 0],
  [-1, 0, 0, 0, 1],
  [-1, 0, 0, 1, 0],
  [-1, 0, 0, 1, 1],
  [-1, 0, 1, 0, 0],
  [-1, 0, 1, 0, 1],
  [-1, 0, 1, 1, 0],
  [-1, 0, 1, 1, 1],

  [-1, -1, 0, 0, 0],
  [-1, -1, 0, 0, 1],
  [-1, -1, 0, 1, 0],
  [-1, -1, 0, 1, 1],
  [-1, -1, 1, 0, 0],
  [-1, -1, 1, 0, 1],
  [-1, -1, 1, 1, 0],
  [-1, -1, 1, 1, 1],

  [1, -1, 0, 0, 0],
  [1, -1, 0, 0, 1],
  [1, -1, 0, 1, 0],
  [1, -1, 0, 1, 1],
  [1, -1, 1, 0, 0],
  [1, -1, 1, 0, 1],
  [1, -1, 1, 1, 0],
  [1, -1, 1, 1, 1],

  [-1, 1, 0, 0, 0],
  [-1, 1, 0, 0, 1],
  [-1, 1, 0, 1, 0],
  [-1, 1, 0, 1, 1],
  [-1, 1, 1, 0, 0],
  [-1, 1, 1, 0, 1],
  [-1, 1, 1, 1, 0],
  [-1, 1, 1, 1, 1]]