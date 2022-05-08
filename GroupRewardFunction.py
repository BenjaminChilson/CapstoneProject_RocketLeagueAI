from functools import total_ordering
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, BallYCoordinateReward, EventReward, FaceBallReward, TouchBallReward
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM
from OurRewardFunction import OurRewardFunction

class GroupRewardFunction(RewardFunction):

    def __init__(self):
        self.function_list = [
            EventReward(goal=1000, concede=-2000, shot=300, touch=100),
            VelocityPlayerToBallReward(),
            VelocityBallToGoalReward()
        ]

    def reset(self, initial_state: GameState):
        for f in self.function_list:
            f.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        total_reward = 0
        for f in self.function_list:
            total_reward += f.get_reward(player, state, previous_action)
        return total_reward


    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        total_reward = 0
        for f in self.function_list:
            total_reward += f.get_final_reward(player, state, previous_action)
        return total_reward
