import os
import numpy as np
import time

def save_training_results_as_csv(training_timestamp, ticks, total_reward, start_time):
    end_time = int(time.time() * 1000.0)
    if not os.path.exists("save/{}/".format(training_timestamp)):
      os.makedirs("save/{}/".format(training_timestamp))
    f = open("save/{}/stats.csv".format(training_timestamp), 'a')
    stat_titles = ['ticks in episode', 'total reward earned in episode', 'start time', 'end time']
    np.savetxt(f, [stat_titles], fmt=''.join(['%s']), delimiter=',')
    stat_values = [ticks, total_reward, start_time, end_time]
    np.savetxt(f, [stat_values], fmt=''.join(['%s']), delimiter=',')
    f.close()