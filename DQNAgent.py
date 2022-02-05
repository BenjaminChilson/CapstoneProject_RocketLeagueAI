import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import os

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
    model.add(Dense(36, input_dim = self.state_size, activation='relu', name="first_hidden_layer"))
    model.add(Dense(50, activation='relu', name="second_hidden_layer"))
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
      state = np.reshape(state, [1, self.state_size])
      predictions = self.model.predict(state)[0]
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

  def save_weight_as_csv(self, directory, filename):
    if not os.path.exists(directory):
      os.makedirs(directory)
    f = open(directory + filename, 'a')
    
    preweights = self.model.get_weights()
    w = 0
    layernumber = 0
    while w < len(preweights):
      layertitle = ["layer" + str(layernumber) + "to" + str(layernumber + 1)]
      np.savetxt(f, [layertitle], fmt=''.join(['%s']), delimiter=',')
      currentLayer = preweights[w]
      i = 0
      for n in currentLayer:
        nodeweights = ["node " + str(i)]
        nodeweights.extend(n)
        i += 1
        np.savetxt(f, [nodeweights], fmt=''.join(['%s']), delimiter=',')
      layer_biases = ["layer " + str(layernumber + 1) + " biases"]
      layer_biases.extend(preweights[w + 1])
      np.savetxt(f, [layer_biases], fmt=''.join(['%s']), delimiter=',')
      layernumber += 1
      w += 2
    f.close