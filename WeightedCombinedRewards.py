from functools import total_ordering
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np
from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward, LiuDistanceBallToGoalReward, VelocityPlayerToBallReward, VelocityBallToGoalReward, BallYCoordinateReward, EventReward, FaceBallReward, TouchBallReward
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM
from OurRewardFunction import OurRewardFunction  

from sb3_log_reward import SB3CombinedLogReward

def combined_reward():
    reward = SB3CombinedLogReward
    (
        (
            EventReward(goal=15, concede=-5, shot=0.1),
            TouchBallReward(1.5),
            LiuDistanceBallToGoalReward(),
            LiuDistancePlayerToBallReward(),
            VelocityBallToGoalReward(),
            VelocityPlayerToBallReward(),
        ),
        (
            1,
            0.4,
            0.2,
            0.1,
            0.5,
            0.2
        )
    )
    return reward