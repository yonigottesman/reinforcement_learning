import numpy as np
import matplotlib.pyplot as plt


class ExpExplorer:
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
        return max(self.explore_rate_stop, explore_probability)


class LinearExplorer:
    def __init__(self, start_e=1, end_e=0.1, steps=1000000, end2_e=0.01, steps2=2000000):
        self.end2 = end2_e
        self.steps = steps
        self.steps2 = steps2

        self.m = (end_e - start_e) / steps
        self.b = start_e

        self.m2 = (self.end2 - end_e) / (steps2-steps)
        self.b2 = end_e - self.m2*self.steps

    def should_explore(self, step):

        exp_exp_trade_off = np.random.rand()
        explore_probability = self.explore_prob(step)
        if explore_probability > exp_exp_trade_off:
            return True
        else:
            return False

    def explore_prob(self, step):
        if step > self.steps2:
            return self.end2
        elif step > self.steps:
            return self.m2 * step + self.b2
        else:
            return self.m*step + self.b


if __name__ == '__main__':
    steps = range(30000000)
    # explorer = LinearExplorer(1, 0.1, 1000000)
    explorer = LinearExplorer(1, 0.1, 1000000, 0.01, 24000000)
    fig, ax = plt.subplots(1, 1)
    ax.plot(steps, [explorer.explore_prob(i) for i in steps])
    plt.show()
