import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import control_states as cs
import full_action_set_constants as fasc

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
      control_state_index = np.random.random_integers(0, self.action_size - 1)
      
      return control_state_index

    #exploitation is chosen
    else:
      updated_state = np.reshape(state, [1, self.state_size])
      predictions = self.model.predict(updated_state)[0]
      control_state_index = np.argmax(predictions)
      
      return control_state_index

  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    for (state, action, reward, next_state, done) in minibatch:
      target = reward

      next_state = np.reshape(next_state, [1, self.state_size])
      state = np.reshape(state, [1, self.state_size])


      if not done:
        target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

      target_f = self.model.predict(state)
      target_f[0][action] = target

      self.model.fit(state, target_f, epochs=1, verbose=0)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)

  def interperet_car_on_ground_shortend_action_set(self, shortened_action_set, interpreted_action_set):
    interpreted_action_set = [None] * 8

    #copy over values for throttle and steer
    interpreted_action_set[fasc.THROTTLE_INDEX] = shortened_action_set[cs.FOWARDBACKWARD_INDEX]
    interpreted_action_set[fasc.STEER_INDEX] = shortened_action_set[cs.LEFTRIGHT_INDEX]
    #yaw, pitch, and roll cannot be adjusted when car is on ground
    interpreted_action_set[fasc.YAW_INDEX] = 0
    interpreted_action_set[fasc.PITCH_INDEX] = 0
    interpreted_action_set[fasc.ROLL_INDEX] = 0
    #copy over values for jump, boost, and handbrake
    interpreted_action_set[fasc.JUMP_INDEX] = shortened_action_set[cs.JUMP_INDEX]
    interpreted_action_set[fasc.BOOST_INDEX] = shortened_action_set[cs.BOOST_INDEX]
    interpreted_action_set[fasc.HANDBRAKE_INDEX] = shortened_action_set[cs.SHIFT_INDEX]

    return interpreted_action_set

  def interperet_car_in_air_shortend_action_set(self, shortened_action_set):
    interpreted_action_set = [None] * 8
    
    roll_activated = shortened_action_set[cs.SHIFT_INDEX] == 1
    roll_not_activated = not roll_activated

    #throttle and steer cannot be adjusted when car is in the air, equals 0
    interpreted_action_set[fasc.THROTTLE_INDEX] = 0
    interpreted_action_set[fasc.STEER_INDEX] = 0
    #copy Forward/Back to control pitch
    interpreted_action_set[fasc.PITCH_INDEX] = shortened_action_set[cs.FOWARDBACKWARD_INDEX]

    if roll_activated:
      #yaw cannot be altered when rolling, equals 0
      interpreted_action_set[fasc.YAW_INDEX] = 0
      #copy Left/Right value to control roll
      interpreted_action_set[fasc.ROLL_INDEX] = shortened_action_set[cs.LEFTRIGHT_INDEX]
    
    elif roll_not_activated:
      #copy Left/Right value to control yaw
      interpreted_action_set[fasc.YAW_INDEX] = shortened_action_set[cs.LEFTRIGHT_INDEX]
      #roll cannot be altered when roll is not activated, equals 0
      interpreted_action_set[fasc.ROLL_INDEX] = 0

    #copy over values for jump and boost
    interpreted_action_set[fasc.JUMP_INDEX] = shortened_action_set[cs.JUMP_INDEX]
    interpreted_action_set[fasc.BOOST_INDEX] = shortened_action_set[cs.BOOST_INDEX]
    #handbrake cannot be used while car is not on ground, equals 0
    interpreted_action_set[fasc.HANDBRAKE_INDEX] = 0

    return interpreted_action_set


  def interperet_control_state(self, control_state, game_state):
    interpreted_control_state = [None] * 8

    car_on_ground = game_state[15]
    car_in_air = not car_on_ground
    
    if car_on_ground:
      interpreted_control_state = self.interperet_car_on_ground_shortend_action_set(control_state, interpreted_control_state)
      
      return interpreted_control_state

    elif car_in_air:
      interpreted_control_state = self.interperet_car_in_air_shortend_action_set(control_state, interpreted_control_state)
      return interpreted_control_state