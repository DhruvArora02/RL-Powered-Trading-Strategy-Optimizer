import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb

def plot_policy(q_table, config):
    env_name, _ = config
    if env_name == 'CliffWalking-v0':
        n_rows, n_cols = 4, 12
        action_to_arrow = {
            0: '↑',  # up
            1: '→',  # right
            2: '↓',  # down
            3: '←'   # left
        }
    
    policy_grid = np.empty((n_rows, n_cols), dtype=object)
    
    for state in range(n_rows * n_cols):
        row = state // n_cols
        col = state % n_cols
        prediction = np.array([q_table[(state,i)] for i in range(4)])
        best_action = np.argmax(prediction)
        policy_grid[row, col] = action_to_arrow[best_action]
    
    if env_name == 'CliffWalking-v0':
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        
        for row in range(n_rows):
            for col in range(n_cols):
                ax.text(col + 0.5, n_rows - row - 0.5, policy_grid[row, col], 
                        ha='center', va='center', fontsize=20)
        
        cliff_area = plt.Rectangle((1, 0), 10, 1, fill=True, color='red', alpha=0.3)
        ax.add_patch(cliff_area)
        
        ax.text(0.5, 0.5, 'S', ha='center', va='center', fontsize=20, fontweight='bold')
        ax.text(n_cols - 0.5, 0.5, 'G', ha='center', va='center', fontsize=20, fontweight='bold')
        
        ax.set_title('Cliff Walking Policy')

        plt.show()

def default_Q_value():
    return 0

def test_RL_agent(config, visualize = False):
    env_name, algo_name = config[0], config[1]
    loaded_data = pickle.load(open(f'Q_TABLE_{algo_name}.pkl', 'rb'))
    Q_table = loaded_data[0]
    EPSILON = loaded_data[1]
    return Q_table

if __name__ == "__main__":
    print('-' * 40)
    config = ('CliffWalking-v0', 'QLearning')
    #config = ('CliffWalking-v0', 'SARSA')
    try:
        Q_table = test_RL_agent(config)
    except Exception as e:
        print(e)
    plot_policy(Q_table, config)
