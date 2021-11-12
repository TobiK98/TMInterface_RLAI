import matplotlib.pyplot as plt


def plot_learning(n_games, score, avg_score, plot_number):
    fig, ax = plt.subplots(figsize=(30, 15))
    ax.set(xlabel='Number of games', ylabel='Score', title='DQN performance in Trackmania')
    ax.plot(n_games, score, 'b-', label='Score')
    ax.plot(n_games, avg_score, 'g-', label='Average Score')
    plt.legend()
    fig.savefig(f'./reinforcement/torch_nn_2/training_result_{plot_number}.png')
    #plt.show()