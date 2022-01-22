import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import short_action_set_constants as sasc
import full_action_set_constants as fasc
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
    interpreted_action_set = [None] * 8

    car_on_ground = state[15]
    car_in_air = not car_on_ground
  
    
    if car_on_ground:
      #copy over values for throttle and steer
      interpreted_action_set[fasc.THROTTLE_INDEX] = shortened_action_set[sasc.FOWARDBACKWARD_INDEX]
      interpreted_action_set[fasc.STEER_INDEX] = shortened_action_set[sasc.LEFTRIGHT_INDEX]
      #yaw, pitch, and roll cannot be adjusted when car is on ground
      interpreted_action_set[fasc.YAW_INDEX] = 0
      interpreted_action_set[fasc.PITCH_INDEX] = 0
      interpreted_action_set[fasc.ROLL_INDEX] = 0
      #copy over values for jump, boost, and handbrake
      interpreted_action_set[fasc.JUMP_INDEX] = shortened_action_set[sasc.JUMP_INDEX]
      interpreted_action_set[fasc.BOOST_INDEX] = shortened_action_set[sasc.BOOST_INDEX]
      interpreted_action_set[fasc.HANDBRAKE_INDEX] = shortened_action_set[sasc.SHIFT_INDEX]
      
      return interpreted_action_set

    elif car_in_air:
      roll_activated = shortened_action_set[sasc.SHIFT_INDEX] == 1
      roll_not_activated = not roll_activated

      #throttle and steer cannot be adjusted when car is in the air, equals 0
      interpreted_action_set[fasc.THROTTLE_INDEX] = 0
      interpreted_action_set[fasc.STEER_INDEX] = 0
      #copy Forward/Back to control pitch
      interpreted_action_set[fasc.PITCH_INDEX] = shortened_action_set[sasc.FOWARDBACKWARD_INDEX]

      if roll_activated:
        #yaw cannot be altered when rolling, equals 0
        interpreted_action_set[fasc.YAW_INDEX] = 0
        #copy Left/Right value to control roll
        interpreted_action_set[fasc.ROLL_INDEX] = shortened_action_set[sasc.LEFTRIGHT_INDEX]
      
      elif roll_not_activated:
        #copy Left/Right value to control yaw
        interpreted_action_set[fasc.YAW_INDEX] = shortened_action_set[sasc.LEFTRIGHT_INDEX]
        #roll cannot be altered when roll is not activated, equals 0
        interpreted_action_set[fasc.ROLL_INDEX] = 0

      #copy over values for jump and boost
      interpreted_action_set[fasc.JUMP_INDEX] = shortened_action_set[sasc.JUMP_INDEX]
      interpreted_action_set[fasc.BOOST_INDEX] = shortened_action_set[sasc.BOOST_INDEX]
      #handbrake cannot be used while car is not on ground, equals 0
      interpreted_action_set[fasc.HANDBRAKE_INDEX] = 0

      return interpreted_action_set

  #throttle, steer, yaw, pitch, roll, jump, boost, handbrake
  def decode_action_set(self, action_set):
    decoded_action_set = [None] * 5
    
    #action_set[0] will be controlling either throttle or pitch
    for x in (action_set[fasc.THROTTLE_INDEX], action_set[fasc.PITCH_INDEX]):
      if x != 0:
        decoded_action_set[sasc.FOWARDBACKWARD_INDEX] = x
        break
    if action_set[fasc.THROTTLE_INDEX] == 0 and action_set[fasc.PITCH_INDEX] == 0:
      decoded_action_set[sasc.FOWARDBACKWARD_INDEX] = 0
    
    #action_set[1] will be controlling either steering, yaw, or rolling
    for x in (action_set[fasc.STEER_INDEX], action_set[fasc.YAW_INDEX], action_set[fasc.ROLL_INDEX]):
      if x != 0:
        decoded_action_set[sasc.LEFTRIGHT_INDEX] = x
        break
    if action_set[fasc.STEER_INDEX] == 0 and action_set[fasc.YAW_INDEX] == 0 and action_set[fasc.ROLL_INDEX] == 0:
      decoded_action_set[sasc.LEFTRIGHT_INDEX] = 0

    decoded_action_set[sasc.JUMP_INDEX] = action_set[fasc.JUMP_INDEX]
    decoded_action_set[sasc.BOOST_INDEX] = action_set[fasc.BOOST_INDEX]
    decoded_action_set[sasc.SHIFT_INDEX] = action_set[fasc.HANDBRAKE_INDEX]

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