import numpy as np
import matplotlib.pyplot as plt

class Explorer():
    def __init__(self, explore_rate_start, explore_rate_stop, decay):
        self.explore_rate_start = explore_rate_start
        self.explore_rate_stop = explore_rate_stop
        self.decay = decay
        self.decay_step = 0

    def explore(self):

        exp_exp_trade_off = np.random.rand()
        explore_probability = self.explore_prob()
        self.decay_step += 1
        if explore_probability > exp_exp_trade_off:
            return True
        else:
            return False

    def explore_prob(self):
        explore_probability = self.explore_rate_stop + (
                self.explore_rate_start - self.explore_rate_stop) * np.exp(
            -self.decay * self.decay_step)
        return explore_probability


if __name__ == '__main__':
    steps = range(1000000)
    exploration = []
    explorer = Explorer(1, 0.01, 0.000001)
    for i in steps:
        exploration.append(explorer.explore_prob())
        explorer.explore()
    fig, ax = plt.subplots(1, 1)
    ax.plot(steps, exploration)
    ax.set_xscale('log')
    plt.show()
