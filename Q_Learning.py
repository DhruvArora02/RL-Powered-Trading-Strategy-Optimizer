import gymnasium as gym
import random
import numpy as np
import time
from collections import deque
import pickle
from collections import defaultdict

EPISODES =  30000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999

def default_Q_value():
    return 0

if __name__ == "__main__":
    env_name = "CliffWalking-v0"
    env = gym.envs.make(env_name)
    env.reset(seed=1)

    Q_table = defaultdict(default_Q_value) 
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()[0]

        while not done:
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()  # Explore: select a random action
            else:
                q_vals = [Q_table[(obs, a)] for a in range(env.action_space.n)]
                max_q = max(q_vals)
                best_actions = [a for a, q in enumerate(q_vals) if q == max_q]
                action = random.choice(best_actions)

            result  = env.step(action)
            next_obs, reward, terminated, truncated, info = result
            done = terminated or truncated

            if not done:
                future_q = max([Q_table[(next_obs, a)] for a in range(env.action_space.n)])
                Q_table[(obs, action)] = (1 - LEARNING_RATE) * Q_table[(obs, action)] + LEARNING_RATE * (reward + DISCOUNT_FACTOR * future_q)
            else:
                Q_table[(obs, action)] = (1 - LEARNING_RATE) * Q_table[(obs, action)] + LEARNING_RATE * reward

            obs = next_obs
            episode_reward = episode_reward + reward

        EPSILON *= EPSILON_DECAY

        episode_reward_record.append(episode_reward) 
     
        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    model_file = open(f'Q_TABLE_QLearning.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()
