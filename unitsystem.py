def density(unit):
    l = unit.to("metre")
    m = unit.to("kilogram")
    return m / l**3


def energy(unit):
    s = unit.to("second")
    l = unit.to("metre")
    m = unit.to("kilogram")
    return m / l**2 / s**2


class Unitsystem:

    def __init__(self, unit_name="earth"):

        import numpy as np

        Rearth = 6371e3
        Mearth = 5.972e24
        Tearth = np.sqrt(Rearth**3 / (Mearth * 6.67e-11))

        if unit_name == "earth":
            self.length = Rearth
            self.time = Tearth
            self.mass = Mearth
        pass

    def to(self, name):
        if name == "metre":
            return self.length
        elif name == "second":
            return self.time
        elif name == "kilogram":
            return self.mass
