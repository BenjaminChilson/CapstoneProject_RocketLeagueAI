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
gym_env = rlgym.make(game_speed=100, obs_builder=OurObsBuilder(), terminal_conditions=[GoalScoredCondition(), NoTouchTimeoutCondition(max_steps=250), TimeoutCondition(1000)], reward_fn=GroupRewardFunction())
env = SB3SingleInstanceEnv(gym_env)

# create the model
model = PPO(
        "MlpPolicy",
        env,
        n_epochs=32,                 # PPO calls for multiple epochs
        learning_rate=1e-5,          # Around this is fairly common for PPO
        ent_coef=0.01,               # From PPO Atari
        vf_coef=1.,                  # From PPO Atari
        gamma=0.995,                 # Gamma as calculated using half-life
        verbose=3,                   # Print out all the info as we're going
        batch_size=32000,            # Batch size as high as possible within reason
        n_steps=32000,               # Number of steps to perform before optimizing network
        tensorboard_log="out/logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
        device="auto"                # Uses GPU if available
    )

# used to save the model after every X amount of steps
save = CheckpointCallback(250000, save_path="policy", name_prefix="CarBallAI")

# start training, always call env.reset() before model.learn()
env.reset()
model.learn(total_timesteps=int(1e6), callback=save, reset_num_timesteps=False)