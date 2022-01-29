import fractions


if __name__ == '__main__':
    def exact(value):
        if isinstance(value, float):
            return fractions.Fraction(value)
        elif isinstance(value, fractions.Fraction):
            return value
        elif isinstance(value, str) or isinstance(value, bytes):
            try:
                text = value.strip()
                integral, _, rem = text.partition('.')
                integral = integral.strip()
                fractional, _, exp = rem.partition('e')
                fractional = fractional.strip()
                exp = exp.strip()
                integral_val = 0 if len(integral) == 0 else int(integral)
                fractional_val = 0 if len(fractional) == 0 else int(fractional)
                base_exp_val = 0 if len(exp) == 0 else int(exp)
                additional_exp = len(fractional)
                if integral_val == 0:
                    leading_zeros = 0
                    for c in fractional:
                        if c == '0':
                            ++leading_zeros
                        else:
                            break
                    additional_exp -= leading_zeros
                exp_val = base_exp_val + additional_exp
                den = 10**exp_val
                num = (10**additional_exp * integral_val) + fractional_val
                return fractions.Fraction(num, den)
            except ValueError as e:
                print(e)
            except Exception as e:
                print(e)

        assert False, 'Error: function exact() undefined for values of type {}'.format(type(value))

    cs = [
        '42.05',
        '3**2',
        'abs(-4.5)',
        'squared(5)',
        'a*b',
        'exact(a)',
        "exact('.0036')",
        "exact('3.1400')",
        "exact('1.045e2')",
        "exact('100')",
        "exact('0.9144')",
        'f*f',
        "exact('0.83612736')",
        'pow(f, 0.5)'
    ]

    gl = {'__builtins__': __builtins__}
    gl['a'] = 0.25
    gl['b'] = 123
    gl['f'] = fractions.Fraction(1143, 1250)
    gl['squared'] = lambda x: x**2
    gl['exact'] = exact
    for c in cs:
        f = '__result__ = {}'.format(c)
        exec(f, gl)
        print('|{:<20}| -> {}'.format(c, gl['__result__']))
