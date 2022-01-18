from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState
import numpy as np

class OurObsBuilder(ObsBuilder):
  def reset(self, initial_state: GameState):
    pass

  def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
    obs = []
    print(state.ball.position.round(3))
    obs.extend(state.ball.position.round(3).tolist())
    obs.extend(state.ball.angular_velocity.tolist())
    print(player.car_data.position.round(3))
    obs.extend(player.car_data.position.tolist())
    obs.extend(player.car_data.angular_velocity.tolist())
    obs.append(player.on_ground)
    
    return np.asarray(obs)