from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils import common_values
import numpy as np

class OurObsBuilder(ObsBuilder):

  CAR_ON_GROUND_INDEX = 15
  STATE_SIZE = 17
  STATE_TITLES = ['BALL POS X', 'BALL POS Y', 'BALL POS Z', 'BALL LINE_VEL X', 'BALL LINE_VEL Y', 'BALL LINE_VEL Z', 'PLAYER POS X', 'PLAYER POS Y', 'PLAYER POS Z', 'PLAYER YAW', 'PLAYER PITCH', 'PLAYER ROLL', 'PLAYER LINE_VEL X', 'PLAYER LINE_VEL Y', 'PLAYER LINE_VEL Z', 'ON GROUND', 'BOOST AMOUNT']

  def reset(self, initial_state: GameState):
    pass

  def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
    obs = []

    inverted = player.team_num == common_values.ORANGE_TEAM

    if not inverted:
      player_orientation = player.car_data
      ball_orientation = state.ball
    else:
      player_orientation = player.inverted_car_data
      ball_orientation = state.inverted_ball

    ball_position = [round(num, 3) for num in ball_orientation.position.tolist()]
    obs.extend(ball_position)
    ball_linear_velocity = [round(num, 3) for num in ball_orientation.linear_velocity.tolist()]
    obs.extend(ball_linear_velocity)
    player_position = [round(num, 3) for num in player_orientation.position.tolist()]
    obs.extend(player_position)
    player_yaw = round(player_orientation.yaw(), 6)
    player_pitch = round(player_orientation.pitch(), 6)
    player_roll = round(player_orientation.roll(), 6)
    obs.extend([player_yaw, player_pitch, player_roll])
    player_linear_velocity = [round(num, 3) for num in player_orientation.linear_velocity.tolist()]
    obs.extend(player_linear_velocity)
    obs.append(round(float(player.on_ground), 3))
    obs.append(player.boost_amount)
    
    return obs