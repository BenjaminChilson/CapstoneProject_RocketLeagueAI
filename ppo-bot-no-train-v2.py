import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions import CombinedReward

import atexit
from torch.nn import Tanh
from sb3_log_reward import SB3CombinedLogReward, SB3CombinedLogRewardCallback


# V1->V2
# increased learning rate
# reduced epochs
# increased net size
# discrete actions instead of continuous
# obs includes other players
# better initial rewards for the first 100M steps
# 1v1 training against itself
# slower transitioning of rewards

if __name__ == '__main__': 
    frame_skip = 8        
    half_life_seconds = 5 

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    num_instances = 1

    # train using the 2 velocity rewards until ~100M and then start using AnnealReward to phase into other rewards
    # need to make sure we don't drastically change rewards too quickly like we did on CarBallAI-V1
    # ... so use AnnealReward
    def get_match():
        return Match(
            team_size=1,
            tick_skip=frame_skip,
            self_play=False,
            obs_builder=AdvancedObs(), 
            state_setter=DefaultState(),
            action_parser=DiscreteAction(),
            game_speed=1,
            reward_function=SB3CombinedLogReward(
            (
                VelocityPlayerToBallReward(),
                VelocityBallToGoalReward(),
                EventReward(
                    team_goal=100.0
                ),
                 EventReward(
                    concede=-100.0
                ),
                EventReward(
                    shot=5.0
                ),
                EventReward(
                    save=30.0
                ),
                EventReward(
                    demo=10.0
                )              
            ),
            (0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
            terminal_conditions=[]
        )

    env = SB3MultipleInstanceEnv(get_match, num_instances)
    env = VecCheckNan(env)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, gamma=gamma)

    model = PPO.load(
            "models/CarBallAI-V2_100000000_steps.zip",
            env,
            device="auto",
            custom_objects=dict(n_envs=env.num_envs))

run = 1
while(run == 1):
    obs = env.reset()
    done = False
    while not done:
        # pass observation to model to get predicted action
        action, _states = model.predict(obs)

        # pass action to env and get info back
        obs, rewards, done, info = env.step(action)
