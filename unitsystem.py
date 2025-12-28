def density(unit):
    l = unit.to("metre")
    m = unit.to("kilogram")
    return m / l**3


def energy(unit):
    l = unit.to("metre")
    s = unit.to("second")
    m = unit.to("kilogram")
    return m * l**2 / s**2


def sp_energy(unit):
    l = unit.to("metre")
    s = unit.to("second")
    return l**2 / s**2


def speed(unit):
    s = unit.to("second")
    l = unit.to("metre")
    return l / s


class Unitsystem:

    def __init__(self, unit_name="earth"):

        import numpy as np

        self.Rearth = 6371e3
        self.Mearth = 5.972e24
        self.Tearth = np.sqrt(self.Rearth**3 / (self.Mearth * 6.67e-11))

        self.Rmoon = 1737e3
        self.Mmoon = 7.347e22
        self.Tmoon = np.sqrt(self.Rmoon**3 / (self.Mmoon * 6.67e-11))

        if unit_name == "earth":
            self.length = self.Rearth
            self.time = self.Tearth
            self.mass = self.Mearth
        elif unit_name == "SI":
            self.length = 1
            self.time = 1
            self.mass = 1
        else:
            raise NotImplementedError()

    def to(self, name):
        if name == "metre":
            return self.length
        elif name == "second":
            return self.time
        elif name == "kilogram":
            return self.mass


if __name__ == "__main__":
    u = Unitsystem("earth")
    print(u.Rearth / u.Tearth)
    print(u.Mearth / u.Rearth**3)
    print("energy", u.Mearth * u.Rearth**2 / u.Tearth**2)
    print("specific energy", u.Rearth**2 / u.Tearth**2)
    print(speed(u))
