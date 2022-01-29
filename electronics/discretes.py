#from theory import Impedance

# TODO: Should these class methods return  components or raw numbers???


class __CircuitComponent:
    __ID = 0

    def __init__(self,  impedance=None):
        self.__impedance = impedance
        self.__id = __CircuitComponent.__ID
        __CircuitComponent.__ID += 1

    @property
    def impedance( self ):
        return self.__impedance  # TODO: Need a way to bind this to changes in the value etc without having to guarantee that the sub-class properly recalculates this.


class __DiscreteComponent(__CircuitComponent):
    @classmethod
    def closest(cls, value, list_type=None):
        target = value
        collection = cls.values(list_type)
        return min((abs(target - i), i) for i in collection)[1]

    @classmethod
    def floor(cls, value, list_type=None):
        target = value
        collection = cls.values(list_type)
        values = [value for value in collection if value <= target]
        if len(values) > 0:
            values.sort()
            return values[-1]
        return None

    @classmethod
    def ceil(cls, value, list_type=None):
        target = value
        collection = cls.values(list_type)
        values = [value for value in collection if value >= target]
        if len(values) > 0:
            values.sort()
            return values[0]
        return None

    @classmethod
    def inRange(cls, value, list_type=None):
        target = value
        collection = cls.values(list_type)
        return min(collection) <= target <= max(collection)

    @classmethod
    def values(cls,  list_type=None):
        return []

    def __init__(self, value=None):
        self.__value = value

    def __float__(self): return float(self.__value)


class __CompoundComponent(__CircuitComponent): pass


class Resistor( __DiscreteComponent ):
    @classmethod
    def values(cls, list_type=None):  # NOTE: This needs to handle 5% resistors also ( as a minimum )
        def resistors_1_percent():
            multipliers = [
                10.0, 	10.2, 	10.5, 	10.7, 	11.0, 	11.3, 	11.5, 	11.8, 	12.1, 	12.4, 	12.7, 	13.0,
                13.3, 	13.7, 	14.0, 	14.3, 	14.7, 	15.0, 	15.4, 	15.8, 	16.2, 	16.5, 	16.9, 	17.4,
                17.8, 	18.2, 	18.7, 	19.1, 	19.6, 	20.0, 	20.5, 	21.0, 	21.5, 	22.1, 	22.6, 	23.2,
                23.7, 	24.3, 	24.9, 	25.5, 	26.1, 	26.7, 	27.4, 	28.0, 	28.7, 	29.4, 	30.1, 	30.9,
                31.6, 	32.4, 	33.2, 	34.0, 	34.8, 	35.7, 	36.5, 	37.4, 	38.3, 	39.2, 	40.2, 	41.2,
                42.2, 	43.2, 	44.2, 	45.3, 	46.4, 	47.5, 	48.7, 	49.9, 	51.1, 	52.3, 	53.6, 	54.9,
                56.2, 	57.6, 	59.0, 	60.4, 	61.9, 	63.4, 	64.9, 	66.5, 	68.1, 	69.8, 	71.5, 	73.2,
                75.0, 	76.8, 	78.7, 	80.6, 	82.5, 	84.5, 	86.6, 	88.7, 	90.9, 	93.1, 	95.3, 	97.6
            ]
            values = [ 1e6,  1.2e6,  1.1e6,  1.3e6,  1.5e6,  1.6e6,  1.8e6,  2.0e6,  2.2e6 ]
            for decade in [pow(10,  y) for y in range(6)]:
                values += [(multiplier * decade) for multiplier in multipliers]
            return values
        return resistors_1_percent()


class Capacitor(__DiscreteComponent):
    @classmethod
    def values(cls, list_type=None):
        def capacitors_standard():
            values = []
            pf_multipliers = [
                1, 1.1,  1.2,  1.3,  1.5,  1.6,  1.8,  2.0,  2.2,  2.4,  2.7,  3.0,
                3.3,  3.6,  3.9,  4.3,  4.7,  5.1, 5.2,  6.2,  6.8,  7.5,  8.1,  9.2
            ]
            for pf_decade in [pow(10,  (y - 12)) for y in range(3)]:
                values += [(multiplier * pf_decade) for multiplier in pf_multipliers]

            uf_multipliers = [
                0.01,  0.015,  0.022,  0.033,  0.047,  0.068
            ]
            for uf_decade in [pow(10, (y - 7)) for y in range(7)]:
                values += [(multiplier * uf_decade) for multiplier in uf_multipliers]
            values += [0.01]
            return values
        return capacitors_standard()


class Inductor(__DiscreteComponent):
    @classmethod
    def values(cls,  list_type=None):
        def std_inductors():
            multiplicands = [1.0,  1.2,  1.5,  1.8, 2.2,  2.7,  3.3,  3.9,  4.7,  5.6,  6.8, 8.2]
            multipliers = [0.01,  0.1,  1.0,  10]
            values = []
            for multiplier in multipliers:
                for multiplicand in multiplicands:
                    values.append(1e-6 * multiplier * multiplicand)
            values.append(100e-6)
            return values
        return std_inductors()


def optimalPotentialDivider(ratio, Rtot=10e3):
    R2 = ratio * Rtot
    R1 = R2 * (1.0-ratio) / ratio
    R2_final = Resistor.closest(R2)
    R1_final = Resistor.closest(R1) if float(R2_final) <= R2 else Resistor.ceil(R1)
    return R1_final, R2_final


if __name__ == '__main__':
    target_value = 94000.0;

    print("Target Value = {:f}".format(target_value))
    print("Closest Value = {:f}".format(Resistor.closest(target_value)))
    print("Floor Value = {:f}".format(Resistor.floor(target_value)))
    print("Ceil Value = {:f}".format(Resistor.ceil(target_value)))

    print('\n1:4 Divider:')
    R1, R2 = optimalPotentialDivider(0.25, target_value)
    print('R1 = {:f}, R2 = {:f}'.format(R1, R2))
