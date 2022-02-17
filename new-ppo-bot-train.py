import rlgym

from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym.utils.reward_functions.combined_reward import CombinedReward
from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward, LiuDistanceBallToGoalReward, VelocityPlayerToBallReward, VelocityBallToGoalReward, BallYCoordinateReward, EventReward, FaceBallReward, TouchBallReward

from rlgym_tools.sb3_utils import SB3SingleInstanceEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan

from sb3_log_reward import SB3CombinedLogRewardCallback, SB3CombinedLogReward

from OurObsBuilder import OurObsBuilder

# create the environment
gym_env = rlgym.make(game_speed=100, obs_builder=OurObsBuilder(), terminal_conditions=[GoalScoredCondition(), NoTouchTimeoutCondition(max_steps=250), TimeoutCondition(1000)], 
reward_fn=SB3CombinedLogReward
   (
       (
           EventReward(goal=15, concede=-15, shot=0.3),
           TouchBallReward(),
           LiuDistanceBallToGoalReward(),
           LiuDistancePlayerToBallReward(),
           VelocityBallToGoalReward(),
           VelocityPlayerToBallReward()
        ),
        (1, 0.25, 0.1, 0.1, 0.3, 0.15)
   ))

env = SB3SingleInstanceEnv(gym_env)
env = VecCheckNan(env)
env = VecMonitor(env)
env = VecNormalize(env, norm_obs=False, gamma=0.995)

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
save = CheckpointCallback(2_500_000, save_path="policy", name_prefix="CarBallAI")
rewardCallback = SB3CombinedLogRewardCallback(reward_names=["event_reward", "player_touch_ball", "liu_distance_ball_to_goal", "liu_distance_player_to_ball", "velocity_ball_to_goal", "velocity_player_to_goal"], file_location="combinedlogfiles")

# start training, always call env.reset() before model.learn()
env.reset()
model.learn(total_timesteps=int(1e8), callback=[save, SB3CombinedLogRewardCallback(reward_names=["event_reward", "player_touch_ball", "liu_distance_ball_to_goal", "liu_distance_player_to_ball", "velocity_ball_to_goal", "velocity_player_to_goal"])], reset_num_timesteps=False)