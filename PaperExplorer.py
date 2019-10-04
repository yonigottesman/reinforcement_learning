import numpy as np
import matplotlib.pyplot as plt

class PaperExplorer():
    def __init__(self):
        self.step = 0
        self.m = -0.0000009
        self.b = 1
        # m = (0.1-1)/(1000000)

    def explore(self):

        exp_exp_trade_off = np.random.rand()
        explore_probability = self.explore_prob()
        self.step += 1
        if explore_probability > exp_exp_trade_off:
            return True
        else:
            return False

    def explore_prob(self):
        if self.step > 1000000:
            return 0.1
        else:
            return self.m*self.step + self.b

if __name__ == '__main__':
    steps = range(10000000)
    exploration = []
    explorer = PaperExplorer()
    for i in steps:
        exploration.append(explorer.explore_prob())
        explorer.explore()
    fig, ax = plt.subplots(1, 1)
    ax.plot(steps, exploration)
    #ax.set_xscale('log')
    plt.show()
