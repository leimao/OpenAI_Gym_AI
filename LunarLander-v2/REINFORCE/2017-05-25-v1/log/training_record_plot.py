import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_record = pd.read_csv('episode_reward_log.txt', sep='\t')
# training_record.head()
fig = plt.figure()
plt.plot(training_record['EPISODE'], training_record['TOTAL_REWARD'], '.-', c = 'blue')
plt.xlabel('Episode')
plt.ylabel('Learning Performance')
fig.savefig('training_record.jpeg', format='jpeg', dpi=300, bbox_inches='tight')