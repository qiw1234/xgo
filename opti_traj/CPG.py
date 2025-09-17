import numpy as np


class cpgBuilder:
    def __init__(self, initpos, gait='trot'):
        self.alpha = 10
        self.miu = 1
        self.b = 50
        self.gait = gait
        self.getGait()
        self.initPos = initpos
        # self.beta = 0.5
        # self.T = 0.5
        # self.phase = [0, np.pi, np.pi, 0]
        # # self.initPos = np.array([0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5])
        # self.initPos = np.array([0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5])

    def hopf_osci(self, y):
        dydt = np.zeros((8, 1))
        for i in range(4):
            r = y[i] ** 2 + y[i + 4] ** 2

            gama = np.pi / (self.beta * self.T * (np.exp(-self.b * y[i + 4]) + 1)) + np.pi / (
                        (1 - self.beta) * self.T * (np.exp(self.b * y[i + 4]) + 1))

            dydt[i] = self.alpha * (self.miu ** 2 - r) * y[i] + gama * y[i + 4]
            dydt[i + 4] = self.alpha * (self.miu ** 2 - r) * y[i + 4] - gama * y[i]

            for j in range(4):
                dydt[i] += np.cos(self.phase[i] - self.phase[j]) * y[j] - np.sin(self.phase[i] - self.phase[j]) * y[
                    i + 4]
                dydt[i + 4] += np.sin(self.phase[i] - self.phase[j]) * y[j] + np.cos(self.phase[i] - self.phase[j]) * y[
                    i + 4]

        return dydt

    def getGait(self):
        if self.gait == 'trot':
            self.beta = 0.7
            self.T = 0.7
            self.phase = [0, np.pi, np.pi, 0]
        if self.gait == 'spacetrot':
            self.beta = 0.5
            self.T = 0.25
            self.phase = [0, np.pi, np.pi, 0]
        elif self.gait == 'pace':
            self.beta = 0.5
            self.T = 0.8
            self.phase = [0, np.pi, 0, np.pi]
        elif self.gait == 'bound':
            self.beta = 0.4
            self.T = 0.3
            self.phase = [0, 0, np.pi, np.pi]
        elif self.gait == 'walk':
            self.beta = 0.75
            self.T = 0.6
            self.phase = [0, np.pi / 2, np.pi, 1.5 * np.pi]


def endEffectorPos_xy(a, p):
    r = a * (6 * p ** 5 - 15 * p ** 4 + 10 * p ** 3 - 0.5)
    return r

def endEffectorPos_xy_spacetrot(a, p):
    scale = 1.3
    if p > (1-1/scale):
        r = a * (6 * (scale*(p-(1-1/scale))) ** 5 -
                 15 * (scale*(p-(1-1/scale))) ** 4 +
                 10 * (scale*(p-(1-1/scale))) ** 3 - 0.5)
    else:
        r = -0.5 * a
    return r


def endEffectorPos_z(h, p):
    r = h * (-64 * p ** 6 + 192 * p ** 5 - 192 * p ** 4 + 64 * p ** 3)
    return r


def endEffectorPos_z_back(h, p):
    if p > 0.8 or p < 0.2:
        r = 0
    else:
        r = h * (-64 * p ** 6 + 192 * p ** 5 - 192 * p ** 4 + 64 * p ** 3)
    return r
