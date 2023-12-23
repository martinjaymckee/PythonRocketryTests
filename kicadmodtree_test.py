from KicadModTree import * 

footprint_name = 'example_footprint'

# Initialization of footprint
mod = Footprint(footprint_name, 2) #FootprintType.THT)
mod.setDescription('An example footprint')
mod.setTags('example')

# General Values
mod.append(Text(type='reference', text='REF**', at=[0, -3], layer='F.SilkS'))
mod.append(Text(type='value', text=footprint_name, at=[1.5, 3], layer='F.Fab'))

# Silkscreen
mod.append(RectLine(start=[-2, -2], end=[5, 2], layer='F.SilkS'))

# Courtyard
mod.append(RectLine(start=[-2.25, -2.25], end=[5.25, 5.25], layer='F.CrtYd'))

# Pads
mod.append(Pad(number=1, type=Pad.TYPE_SMT, shape=Pad.SHAPE_ROUNDRECT, radius_ratio=.1, at=[0, 0], size=[2, 2], layers=['F.Cu', 'F.Mask']))
mod.append(Pad(number=2, type=Pad.TYPE_THT, shape=Pad.SHAPE_RECT, at=[3, 0], size=[2, 2], drill=1.2, layers=Pad.LAYERS_THT))

# Add model
#mod.append(kmt.Model(filename="example.3dshapes/example_footprint.wrl", at=[0, 0, 0], scale=[1, 1, 1], rotate=[0, 0, 0]))

file_handler = KicadFileHandler(mod)
print(file_handler.serialize())

filename = 'example.kicad_mod'
file_handler.writeFile(filename)