# This is module of potential energy
import numpy as np

def potentialEnergy(x, a=100, c=10, f_ext=1):

    l_energy = 3 - 13*c/a
    l_energy += (c/(2*a) - 1)*np.cos(2*x)
    l_energy += (15*np.sqrt(3)*c/(2*a))*np.sin(2*x)
    l_energy += f_ext*(-(2*a+5*c)*np.cos(x) + 3*np.sqrt(3)*c*np.sin(x))

    return l_energy

def gradEnergy(x, a=100, c=10, f_ext=1):
    l_grad = (2 - c/a) * np.sin(2*x)
    l_grad += (15*np.sqrt(3)*c/a) * np.cos(2*x)
    l_grad += f_ext * ((2*a + 5*c)*np.sin(x) + 3*np.sqrt(3)*c*np.cos(x))

    return l_grad

#-----------------------------------------------------
def potentialEnergySepa(x, a=100, c=10, f_ext=1):
    l_energy = extEnergy(x, a, f_ext)
    l_energy += dipoleEnergy(x)
    l_energy += paraEnergy(x, a, c, f_ext)

    return l_energy

def extEnergy(x, a=100, f_ext=1):
    return -2 * a * f_ext * np.cos(x)

def dipoleEnergy(x):
    return 3 - np.cos(2*x)

def paraEnergy(x, a=100, c=10, f_ext=1):
    perm_field = permanentMagneticField(x, a)
    return -2 * c * (f_ext + perm_field) * (3*np.cos(x + np.pi/3) + np.cos(x))
    #return -2 * c * (f_ext) * (3*np.cos(x + np.pi/3) + np.cos(x))

def permanentMagneticField(x, a=100):
    return -(3*np.sqrt(3)*np.sin(x - np.pi/3) + 2*np.cos(x))/a


def gradEnergySepa(x, a=100, c=10, f_ext=1):
    l_grad = gradExtEnergy(x, a, f_ext)
    l_grad += gradDipoleEnergy(x)
    l_grad += gradParaEnergy(x, a, c, f_ext)

    return l_grad

def gradExtEnergy(x, a=100, f_ext=1):
    return 2 * a * f_ext * np.sin(x)

def gradDipoleEnergy(x):
    return 2 * np.sin(2*x)

def gradParaEnergy(x, a=100, c=10, f_ext=1):
    return 2*(gradExtPara(x, c, f_ext) + gradDipolePara(x, a, c))
    #return 2*gradExtPara(x, c, f_ext)

def gradExtPara(x, c=10, f_ext=1):
    return c*f_ext*(3*np.sin(x + np.pi/3) + np.sin(x))

def gradDipolePara(x, a=100, c=10):
    return (c/a) * ( (3*np.sqrt(3)*np.cos(x - np.pi/3) - 2*np.sin(x))*(3*np.cos(x + np.pi/3) + np.cos(x)) - (3*np.sqrt(3)*np.sin(x - np.pi/3) + 2*np.cos(x))*(3*np.sin(x + np.pi/3) + np.sin(x)) )

#------------------------------------------


def GradientDescent(x0, a=100, c=10, f_ext=1):
    x = x0
    counter = 0
    while(True):
        rate = 0.001 / (1 + counter)
        delta = 0.001*gradEnergy(x, a, c, f_ext)
        if abs(delta) < 1.0e-5: break
        x -= delta
        counter += 1;

    return x

def GradientDescentSepa(x0, a=100, c=10, f_ext=1):
    x = x0
    counter = 0
    while(True):
        rate = 0.001 / (1 + counter)
        delta = 0.001*gradEnergySepa(x, a, c, f_ext)
        #print("delta = {}".format(delta))
        if abs(delta) < 1.0e-4: break
        x -= delta
        counter += 1;

    return x
