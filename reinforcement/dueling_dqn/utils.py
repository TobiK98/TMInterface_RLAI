import matplotlib.pyplot as plt


def plot_learning(n_games, score, avg_score, epsilon, plot_number):
    fig, ax1 = plt.subplots(figsize=(30, 15))
    ax1.set(xlabel='Number of games', ylabel='Score', title='Dueling DQN performance in Trackmania')
    ax2 = ax1.twinx()
    ax2.set(ylabel='Epsilon')
    plot_score = ax1.plot(n_games, score, 'b-', linewidth=1, label='Score')
    plot_avg_score = ax1.plot(n_games, avg_score, 'g-', linewidth=5, label='Average Score')
    plot_epsilon = ax2.plot(n_games, epsilon, 'r-', linewidth=2, label='Epsilon')
    lns = plot_score + plot_avg_score + plot_epsilon
    labs = [i.get_label() for i in lns]
    plt.legend(lns, labs, loc=0)
    fig.savefig(f'./reinforcement/dueling_dqn/training_result_{plot_number}.png')
    # plt.show()