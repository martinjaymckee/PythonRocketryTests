import KicadModTree as kmt  

class TickGenerator:
    def __init__(self):
        pass

    def __call__(self, N, stride, levels, shrink):
        assert False, 'Error: {}.__call__(N, height, shrink) is UNIMPLEMENTED'.format(self.__class__.__name__)


class BinaryTickGenerator(TickGenerator):
    def __init__(self, alternate_lowest=False):
        TickGenerator.__init__(self)
        self.__alternate_lowest = alternate_lowest

    def __call__(self, N, stride, levels, shrink, log_shrink=True):
        stride = int(stride) # TODO: NEED TO CHECK THAT THE STRIDE IS EVENLY DIVISIBLE BY THE REQUESTED NUMBER OF LEVELS
        working_list = [(False, 1)] * N
        level_shrink = 1
        level_stride = stride
        print('stride = {}, shrink = {}, log_shrink = {}'.format(stride, shrink, log_shrink))

        if shrink is None:
            shrink = 0.75 if log_shrink else 1 / (levels+(3 if self.__alternate_lowest else 2))

        print('stride = {}, shrink = {}, log_shrink = {}'.format(stride, shrink, log_shrink))

        for level in range(levels):
            for idx in range(0, N, level_stride):
                if not working_list[idx][0]:
                    working_list[idx] = (True, level_shrink)
            level_stride = int(level_stride / 2)
            level_shrink = level_shrink * shrink if log_shrink else level_shrink - shrink

        even_shrink = level_shrink
        odd_shrink = (even_shrink * shrink if log_shrink else even_shrink - shrink) if self.__alternate_lowest else even_shrink
        for idx in range(N):
            if not working_list[idx][0]:
                working_list[idx] = (True, even_shrink if (idx%2 == 0) else odd_shrink)
        return [tick[1] for tick in working_list]


class RulerGenerator:
    __tick_generators = {
        'binary' : BinaryTickGenerator(),
        'binary_alternating' : BinaryTickGenerator(alternate_lowest=True)
    }

    __unit_multipliers = {
        'mm': 1,
        'in': 25.4
    }

    __fmts = {
        'mm': {
            'ref_unit': 'mm',
            'spacing': 0.5,
            'rep_length': 10,
            'tick_levels': 2
        },
        'in': {
            'ref_unit': 'in',
            'spacing': 0.03125,
            'rep_length': 1,
            'tick_levels': 4
        }
    }

    def __init__(self):
        pass

    def __call__(self, name, length, fmt, filename_fmt=None, **kwargs):
        # Process Formatting
        if isinstance(fmt, str):
            fmt = RulerGenerator.__fmts[fmt]

        if 'ref_unit' in kwargs:
            fmt['ref_unit'] = kwargs['ref_unit']

        if 'spacing' in kwargs:
            fmt['spacing'] = kwargs['spacing']

        if 'height' in kwargs:
            fmt['height'] = kwargs['height']

        if 'padding' in kwargs:
            fmt['padding'] = kwargs['padding']

        if 'tick_levels' in kwargs:
            fmt['tick_levels'] = kwargs['tick_levels']

        if 'tick_shrink' in kwargs:
            fmt['tick_shrink'] = kwargs['tick_shrink']

        if 'tick_log_shrink' in kwargs:
            fmt['tick_log_shrink'] = kwargs['tick_log_shrink']

        if 'tick_generator' in kwargs:
            fmt['tick_generator'] = kwargs['tick_generator']

        number_pads = True
        ref_units = fmt['ref_unit']
        unit_multiplier = self.__unit_multipliers[ref_units]
        tgt_length = length * unit_multiplier
        spacing = fmt['spacing'] * unit_multiplier
        rep_length = fmt['rep_length'] * unit_multiplier
        num_marks = int((tgt_length / spacing) + 0.5) + 1
        print('length = {}, spacing = {}, rep_length = {}, num_marks = {}'.format(tgt_length, spacing, rep_length, num_marks))
        stride_marks = int(rep_length / spacing)
        actual_length = spacing * num_marks
        height = (rep_length / 2 if 'height' not in fmt else fmt['height'] * unit_multiplier)
        padding = (0 if 'padding' not in fmt else fmt['padding']) * unit_multiplier
        tick_levels = 4 if 'tick_levels' not in fmt else fmt['tick_levels']
        tick_shrink = None if 'tick_shrink' not in fmt else fmt['tick_shrink']
        tick_log_shrink = True if 'tick_log_shrink' not in fmt else fmt['tick_log_shrink']
        tick_generator = 'binary_alternating'
        if 'tick_generator' in fmt:
            tick_generator = fmt['tick_generator']
        if isinstance(tick_generator, str):
            tick_generator = RulerGenerator.__tick_generators[tick_generator]
        tick_width = 0.4 * spacing


        # Initialization of Footprint Module
        mod = kmt.Footprint(name, 1) #FootprintType.SMD)
        mod.setDescription('A ruler that is {} {} long with {} {}'.format(length, ref_units, spacing, ref_units))
#       mod.setTags('example')
#       mod.append(Text(type='reference', text='REF**', at=[0, -3], layer='F.SilkS'))
        ruler_name = 'Ruler {} {}'.format(length, ref_units)
        mod.append(kmt.Text(type='value', text=ruler_name, at=[1.5, 3], layer='F.Fab'))

        # Silkscreen
#        mod.append(RectLine(start=[-2, -2], end=[5, 2], layer='F.SilkS'))

        # Courtyard

        mod.append(kmt.RectLine(start=[-tick_width - padding, -padding], end=[actual_length+(tick_width/2)+padding, height+padding], layer='F.CrtYd'))

        # Pads
        pad_height_ratios = tick_generator(num_marks, stride_marks, tick_levels, tick_shrink, log_shrink=tick_log_shrink) 
#        mod.append(Pad(number=1, type=Pad.TYPE_SMT, shape=Pad.SHAPE_ROUNDRECT, radius_ratio=.1, at=[0, 0], size=[2, 2], layers=['F.Cu', 'F.Mask']))
#        mod.append(Pad(number=2, type=Pad.TYPE_THT, shape=Pad.SHAPE_RECT, at=[3, 0], size=[2, 2], drill=1.2, layers=Pad.LAYERS_THT))

        for idx, ratio in enumerate(pad_height_ratios):
            x = idx * spacing
            h = height * ratio
            y = h / 2
            pad_num = idx if number_pads else 1
            mod.append(kmt.Pad(number=idx, type=kmt.Pad.TYPE_SMT, shape=kmt.Pad.SHAPE_ROUNDRECT, radius_ratio=.1, at=[x, y], size=[tick_width, h], layers=['F.Cu', 'F.Mask']))
#            print('x = {}, y = {}, w = {}, h = {}, height={}'.format(x, y, pad_width, h, height))

        if filename_fmt is not None:
            filename = filename_fmt.format(name=name) # TODO: CREATE A DICTIONARY WITH NAME, DATE, TIME, LENGTH, ETC.
            file_handler.writeFile(filename)

        return mod


if __name__ == '__main__':
    generator = RulerGenerator()
    name = '6in_ruler_v1'
#    name = '153mm_ruler_v1'

    mod = generator(name, 6, 'in', height=.25)
#    mod = generator(name, 153, 'mm', height=6)
    file_handler = kmt.KicadFileHandler(mod)
  #  print(file_handler.serialize())

    filename = '{}.kicad_mod'.format(name)
    file_handler.writeFile(filename)