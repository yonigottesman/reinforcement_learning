import numpy as np
import matplotlib.pyplot as plt

class PaperExplorer():
    def __init__(self, start=1, end=0.1, steps=1000000):
        self.step = 0
        self.end = end
        self.start = start
        self.steps = steps
        self.m = (self.end - self.start) / (steps)
        self.b = 1


    def explore(self):

        exp_exp_trade_off = np.random.rand()
        explore_probability = self.explore_prob()
        self.step += 1
        if explore_probability > exp_exp_trade_off:
            return True
        else:
            return False

    def explore_prob(self):
        if self.step > self.steps:
            return self.end
        else:
            return self.m*self.step + self.b

if __name__ == '__main__':
    steps = range(1000000)
    exploration = []
    #explorer = PaperExplorer()
    explorer = PaperExplorer(1, 0.02, 100000)
    for i in steps:
        exploration.append(explorer.explore_prob())
        explorer.explore()
    fig, ax = plt.subplots(1, 1)
    ax.plot(steps, exploration)
    #ax.set_xscale('log')
    plt.show()
