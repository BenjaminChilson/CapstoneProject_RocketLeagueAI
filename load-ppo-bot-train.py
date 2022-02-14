import rlgym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan

from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward, BallYCoordinateReward, EventReward, RewardIfBehindBall
from rlgym.utils.reward_functions.combined_reward import CombinedReward

from rlgym_tools.sb3_utils import SB3SingleInstanceEnv

from OurObsBuilder import OurObsBuilder
from GroupRewardFunction import GroupRewardFunction

# create the environment
gym_env = rlgym.make(game_speed=100, obs_builder=OurObsBuilder(), terminal_conditions=[GoalScoredCondition(), NoTouchTimeoutCondition(max_steps=250), TimeoutCondition(1000)], reward_fn=GroupRewardFunction())
env = SB3SingleInstanceEnv(gym_env)
env = VecCheckNan(env)
env = VecMonitor(env)
env = VecNormalize(env, norm_obs=False, gamma=0.995)

# load the model
model = PPO.load("policy/CarBallAI_100000000_steps.zip", env, device="auto", custom_objects=dict(n_envs=env.num_envs))

# used to save the model after every X amount of steps
save = CheckpointCallback(5000000, save_path="policy", name_prefix="CarBallAI")

# start training, always call env.reset() before model.learn()
env.reset()
model.learn(total_timesteps=int(1e8), callback=save, reset_num_timesteps=False)
