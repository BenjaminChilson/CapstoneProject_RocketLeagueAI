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
from rlgym_tools.extra_rewards.anneal_rewards import AnnealRewards
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
    agents_per_match = 2
    num_instances = 1
    target_steps = 100_000
    steps = target_steps // (num_instances * agents_per_match)
    batch_size = steps

    def exit_save(model):
        model.save("models/exit_save")

    # train using the 2 velocity rewards until ~100M and then start using AnnealReward to phase into other rewards
    # need to make sure we don't drastically change rewards too quickly like we did on CarBallAI-V1
    # ... so use AnnealReward
    def get_match():
        return Match(
            team_size=1,
            tick_skip=frame_skip,
            reward_function=AnnealRewards
            (
                SB3CombinedLogReward
                (
                    (
                        VelocityPlayerToBallReward(),
                        VelocityBallToGoalReward(),
                        EventReward
                        (
                            team_goal=100.0
                        ),
                        EventReward
                        (
                            concede=-100.0
                        ),
                        EventReward
                        (
                            shot=5.0
                        ),
                        EventReward
                        (
                            save=30.0
                        ),
                        EventReward
                        (
                            demo=10.0
                        )              
                    ),
                        (0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
                ),
                150_000_000, # phase out this reward between 100M to 150M. by 150m steps the second reward function is the only one in use
                SB3CombinedLogReward
                (
                    (
                        VelocityPlayerToBallReward(use_scalar_projection=True),
                        VelocityBallToGoalReward(),
                        EventReward
                        (
                            team_goal=100.0
                        ),
                        EventReward
                        (
                            concede=-100.0
                        ),
                        EventReward
                        (
                            shot=5.0
                        ),
                        EventReward
                        (
                            save=30.0
                        ),
                        EventReward
                        (
                            demo=10.0
                        ),
                        EventReward
                        (
                            touch=0.2
                        )                
                    ),
                        (0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                        file_location="annealedlogfiles"
                )
            ),
            self_play=True,
            terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 20), GoalScoredCondition()],
            obs_builder=AdvancedObs(), 
            state_setter=DefaultState(),
            action_parser=DiscreteAction()
        )

    env = SB3MultipleInstanceEnv(get_match, num_instances)
    env = VecCheckNan(env)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, gamma=gamma)

    # load an existing model, or create a new one it doesn't exist
    try:
        print("Looking for existing model ...")
        model = PPO.load(
            "models/exit_save.zip",
            env,
            device="auto",
            custom_objects=dict(n_envs=env.num_envs, n_epochs=10)
        )
        print("Existing model found.")
    except:
        from torch.nn import Tanh
        print("Creating new model ...")
        policy_kwargs = dict(
            activation_fn=Tanh,
            net_arch=[512, 512, dict(pi=[256, 256, 256], vf=[256, 256, 256])],
        )

        model = PPO(
            MlpPolicy,
            env,
            n_epochs=10,                 # PPO calls for multiple epochs
            policy_kwargs=policy_kwargs,
            learning_rate=5e-5,          # Around this is fairly common for PPO
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            verbose=3,                   # Print out all the info as we're going
            batch_size=batch_size,             # Batch size as high as possible within reason
            n_steps=steps,                # Number of steps to perform before optimizing network
            tensorboard_log="logs",  # `tensorboard --logdir logs/logs` in terminal to see graphs
            device="auto"                # Uses GPU if available
        )

    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    callback = CheckpointCallback(round(2_500_000 / env.num_envs), save_path="models", name_prefix="CarBallAI-V2")
    rewardCallback = SB3CombinedLogRewardCallback(reward_names=["velocity_player_to_ball", "velocity_ball_to_goal", "team_goal", "opponent_goal", "shot", "save", "demolition"])
    annealedrewardCallback = SB3CombinedLogRewardCallback(reward_names=["velocity_player_to_ball", "velocity_ball_to_goal", "team_goal", "opponent_goal", "shot", "save", "demolition", "ball_touch"], file_location="annealedlogfiles")


    atexit.register(exit_save, model)
    try:
        while True:
            model.learn(25_000_000, callback=[callback, rewardCallback, annealedrewardCallback], reset_num_timesteps=False)
            model.save("models/exit_save")
            model.save(f"mmr_models/{model.num_timesteps}")
    except Exception as e:
        print(e)
