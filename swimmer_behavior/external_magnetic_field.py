import numpy as np

class ExternalMagneticField:
    def __init__(self, angle=0, angle_velocity=0):
        self.psi = angle
        self.moment = np.array([0, np.cos(self.psi), 0])
        self.omega = angle_velocity

    def update(self, dt):
        self.psi += self.omega * dt
        self.moment = np.array([0, np.cos(self.psi), 0])
