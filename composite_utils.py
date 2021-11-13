
def layupProperties(area, stackup, resin_fraction=0.4):
    total_thickness = 0.0
    total_weight = 0.0

    for count, (weight, thickness) in stackup:
        total_thickness += count * thickness
        total_weight += count * area * weight
    total_weight = total_weight / (1 - resin_fraction)
    return total_thickness, total_weight, total_weight * resin_fraction


bondo_gf = (205, .26)  # gsm, mm
seven_five_gf = (246, .272)
two_gf = (81, .089)
four_gf = (136, .2)
six_gf = (198, .25)

if __name__ == '__main__':
    area = 622 / (100 * 100)  # cm^2 to m^2
    stackup = [(1, four_gf), (2, bondo_gf)]
    print('thickness = {} mm, final weight = {} g, resin weight = {} g'.format(*layupProperties(area, stackup)))
