

def a():
    dsl.append('a')

def b():
    dsl.append('b')

def cb():
    dsl.append('c')
    b()

def asm(f):
    dsl = []
    gl = {'dsl': dsl, 'a': a, 'b': b, 'cb': cb}
    old_gl = f.__globals__.copy()
    f.__globals__.clear()
    f.__globals__.update(gl)

    f()

    f.__globals__.clear()
    f.__globals__.update(old_gl)

    def g():
        return dsl
    return g


if __name__ == '__main__':
    @asm
    def test():
        a()
        b()
        cb()

    print(test())
