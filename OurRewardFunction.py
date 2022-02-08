from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward, BallYCoordinateReward, EventReward
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM

class OurRewardFunction(RewardFunction):

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM and player.car_data.position[1] < state.ball.position[1] \
            or player.team_num == ORANGE_TEAM and player.car_data.position[1] > state.ball.position[1]:
                return 10
        else:
            return 0
