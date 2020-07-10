import numpy as np

class Swimmer:
    alpha = 1.0e+2
    beta = 1.0e-1
    gamma = 1.0e+1
    a_l = 0.3
    def __init__(self, position, init_angle, flag):
        self.flag = flag
        self.pos = position
        self.theta = init_angle
        self.permanent_moment = np.array([-np.sin(self.theta), np.cos(self.theta), 0.0])
        self.para_moment = np.zeros(3)
        self.torque = np.zeros(3)

        self.nx = np.array([1, 0, 0])
        self.npara = np.array([np.cos(np.pi/3), np.sin(np.pi/3), 0])
        self.npara2 = np.array([-np.cos(np.pi/3), np.sin(np.pi/3), 0])

    def calcParamagneticMoment(self, ext_field):
        the_other_moment = np.array([
            -self.permanent_moment[0],
            self.permanent_moment[1],
            self.permanent_moment[2]
            ])
        b_p = 3*np.dot(self.permanent_moment, -self.npara)*(-self.npara) - self.permanent_moment
        b_p += 3*np.dot(the_other_moment, -self.npara2)*(-self.npara2) - the_other_moment

        if self.flag == False:
            self.para_moment = Swimmer.gamma * ext_field
        elif self.flag == True:
            self.para_moment = Swimmer.gamma * (ext_field + b_p/Swimmer.alpha)

    def calcTorque(self, ext_field):
        b_all = Swimmer.alpha * ext_field
        the_other_moment = np.array([
            -self.permanent_moment[0],
            self.permanent_moment[1],
            self.permanent_moment[2]
            ])
        print("in calcTorque()")
        print(b_all)
        b_all += 3*np.dot(the_other_moment, self.nx)*self.nx - the_other_moment
        print(b_all)
        b_all += 3*np.dot(self.para_moment, self.npara)*self.npara - self.para_moment
        print(b_all)

        self.torque = np.cross(self.permanent_moment, b_all)

    def update(self, dt):
        self.theta += Swimmer.beta * ( 1/(Swimmer.a_l**3) + 1/2 ) * self.torque[2] * dt
        self.permanent_moment = np.array([-np.sin(self.theta), np.cos(self.theta), 0.0])

    def particlePosition(self):
        return np.array([
            self.pos - 0.5*self.nx,
            self.pos + 0.5*self.nx,
            self.pos + np.array([0, np.sin(np.pi/3), 0])
            ]).T

    def particleMoment(self):
        the_other_moment = np.array([
            -self.permanent_moment[0],
            self.permanent_moment[1],
            self.permanent_moment[2]
            ])
        return np.array([
            self.permanent_moment,
            the_other_moment,
            self.para_moment/Swimmer.gamma
            ]).T

    def potentialEnergy(self, x_arr, ext_field):
        theta_potential = self.extEnergy(self.theta, ext_field) \
            + self.dipoleEnergy(self.theta)

        if type(x_arr) == np.ndarray:
            x_potential = []
            for x in x_arr:
                x_potential.append(self.extEnergy(x, ext_field) \
                    + self.dipoleEnergy(x))
                #x_potential.append(self.extEnergy(x, ext_field))
        else:
            x_potential = self.extEnergy(x_arr, ext_field) \
                    + self.dipoleEnergy(x_arr)
            #x_potential = self.extEnergy(x_arr, ext_field)

        return theta_potential, x_potential

    def extEnergy(self, theta, ext_field):
        tmp_moment = np.array([
            -np.sin(theta),
            np.cos(theta),
            0
            ])
        field = ext_field.copy()

        if self.flag == 0:
            field += (Swimmer.gamma/Swimmer.alpha) * ext_field[1] * np.array([3*np.sqrt(3)/4, 5/4, 0])
        elif self.flag == 1:
            field += (Swimmer.gamma/Swimmer.alpha) * (ext_field[1] - (3*np.sqrt(3)*np.sin(theta - np.pi/3) + 2*np.cos(theta))/Swimmer.alpha) * np.array([3*np.sqrt(3)/4, 5/4, 0])

        ext_energy = -4 * Swimmer.alpha * np.dot(tmp_moment, field)
        return ext_energy

    def dipoleEnergy(self, theta):
        dipole_energy = 3 - np.cos(2*theta)
        return dipole_energy

    def gradientDescent(self, x0, ext_field):
        x = x0
        while True:
            delta = self.gradEnergy(x, ext_field)
            if abs(delta) < 1.0e-5: break
            x -= 1.0e-4 * delta

        return x

    def gradEnergy(self, theta, ext_field):
        if self.flag == 0:
            grad = self.gradExtEnergy(theta, ext_field) + self.gradDipoleEnergy(theta)
        elif self.flag == 1:
            grad = self.gradExtEnergy2(theta, ext_field) + self.gradDipoleEnergy(theta)

        return grad

    #超常磁性に永久磁石の磁場を含まない場合
    def gradExtEnergy(self, theta, ext_field):
        tmp_vector = np.array([
            -np.cos(theta),
            -np.sin(theta),
            0
            ])
        field = ext_field.copy()
        field += (Swimmer.gamma/Swimmer.alpha) * ext_field[1] * np.array([3*np.sqrt(3)/4, 5/4, 0])
        grad_ext_energy = -4 * Swimmer.alpha * np.dot(tmp_vector, field)
        return grad_ext_energy

    def gradExtEnergy2(self, theta, ext_field):
        tmp_moment = np.array([
            -np.sin(theta),
            np.cos(theta),
            0
            ])
        d_tmp_moment = np.array([
            -np.cos(theta),
            -np.sin(theta),
            0
            ])
        field = ext_field.copy()
        field += (Swimmer.gamma/Swimmer.alpha) * (ext_field[1] - (3*np.sqrt(3)*np.sin(theta - np.pi/3) + 2*np.cos(theta))/Swimmer.alpha) * np.array([3*np.sqrt(3)/4, 5/4, 0])

        d_field = -(Swimmer.gamma/(Swimmer.alpha**2)) * (3*np.sqrt(3)*np.cos(theta - np.pi/3) - 2*np.sin(theta)) * np.array([3*np.sqrt(3)/4, 5/4, 0])

        grad_ext_energy = -4*Swimmer.alpha * (np.dot(tmp_moment, d_field) + np.dot(d_tmp_moment, field))
        return grad_ext_energy

    def gradDipoleEnergy(self, theta):
        grad_dipole_energy = 2 * np.sin(2*theta)
        return grad_dipole_energy
