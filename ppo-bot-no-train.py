import rlgym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward, BallYCoordinateReward, EventReward, RewardIfBehindBall
from rlgym.utils.reward_functions.combined_reward import CombinedReward

from rlgym_tools.sb3_utils import SB3SingleInstanceEnv

from OurObsBuilder import OurObsBuilder
from GroupRewardFunction import GroupRewardFunction

# create the environment
gym_env = rlgym.make(game_speed=1, obs_builder=OurObsBuilder(), terminal_conditions=[GoalScoredCondition(), NoTouchTimeoutCondition(max_steps=250), TimeoutCondition(1000)], reward_fn=GroupRewardFunction())
env = SB3SingleInstanceEnv(gym_env)

# load the model
model = PPO.load("policy/CarBallAI_100000000_steps.zip", env, device="auto", custom_objects=dict(n_envs=env.num_envs))

episodes = 1000
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        # pass observation to model to get predicted action
        action, _states = model.predict(obs)

        # pass action to env and get info back
        obs, rewards, done, info = env.step(action)