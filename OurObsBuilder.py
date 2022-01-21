from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState
import numpy as np

class OurObsBuilder(ObsBuilder):
  def reset(self, initial_state: GameState):
    pass

  def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
    obs = []
    ball_position = [round(num, 3) for num in state.ball.position.tolist()]
    obs.extend(ball_position)
    ball_linear_velocity = [round(num, 3) for num in state.ball.linear_velocity.tolist()]
    obs.extend(ball_linear_velocity)
    player_position = [round(num, 3) for num in player.car_data.position.tolist()]
    obs.extend(player_position)
    player_yaw = round(player.car_data.yaw(), 6)
    player_pitch = round(player.car_data.pitch(), 6)
    player_roll = round(player.car_data.roll(), 6)
    obs.extend([player_yaw, player_pitch, player_roll])
    player_linear_velocity = [round(num, 3) for num in player.car_data.linear_velocity.tolist()]
    obs.extend(player_linear_velocity)
    obs.append(round(float(player.on_ground), 3))
    
    return obs