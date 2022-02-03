import os
import numpy as np
from datetime import datetime

def save_training_results_as_csv(training_timestamp, episode_number, ticks, total_reward, start_time):
    end_time = datetime.now().strftime("%Y:%m:%d_%H:%M:%S")
    if not os.path.exists("save/{}/episode/{}/".format(training_timestamp, episode_number)):
      os.makedirs("save/{}/episode/{}/".format(training_timestamp, episode_number))
    f = open("save/{}/episode/{}/stats.csv".format(training_timestamp, episode_number), 'a')
    stat_titles = ['ticks in episode', 'total reward earned in episode', 'start time', 'end time']
    np.savetxt(f, [stat_titles], fmt=''.join(['%s']), delimiter=',')
    stat_values = [ticks, total_reward, start_time, end_time]
    np.savetxt(f, [stat_values], fmt=''.join(['%s']), delimiter=',')
    f.close()