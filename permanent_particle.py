import numpy as np

class PermanentParticle():
    def __init__(self, pos, moment):
        self.pos = pos
        self.moment = moment

    def printPosition(self):
        print(self.pos)

class ExternalMagneticField():
    def __init__


pos_arr = np.array([0, 1])

perm = PermanentParticle(pos_arr, 0)
perm.printPosition()
