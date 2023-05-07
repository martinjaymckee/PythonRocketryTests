class Matrix:
    @classmethod
    @micropython.native
    def Eye(cls, N, dtype=lambda x: x):
        m = cls()
        for row in range(N):
            m.append_row(*[dtype(1 if r == row else 0) for r in range(N)])
        return m
    
    @classmethod
    @micropython.native
    def Count(cls, N, M=None, init=1, dtype=lambda x: x):
        M = N if M is None else M
        m = cls()
        m.__shape = (N, M)
        m.__data = [dtype(v) for v in range(init, N*M+init)]
        return m

    @classmethod
    @micropython.native
    def SecondDifference(cls, N, dtype=lambda x: x):
        m = cls()
        m.__shape = (N, N)
        data = []
        for row in range(N):
            for column in range(N):
                v = 0
                if row == 0:
                    if column == 0:
                        v = 1
                    elif column == 1:
                        v = -1
                elif row == N-1:
                    if column == N-2:
                        v = 1
                    elif column == N-1:
                        v = -1
                else:
                    if column == row-1:
                        v = 1
                    elif column == row+1:
                        v = -1
                    elif column == row:
                        v = 2
                data.append(dtype(v))
        m.__data = data
        return m

    @classmethod
    @micropython.native
    def Invertible(cls, N, seed=1, dtype=lambda x: x):
        if N < 3:
            cls.Eye(N, dtype=dtype)
        return cls.SecondDifference(N, dtype=dtype)
    
#     class RowIterator:
#         def __init__(self, obj):           
#             self.__obj = obj
#             self.__row = 0
#             
#         def __next__(self):
#             row = self.__row
#             if row < self.__obj.__shape[0]:
#                 self.__row += 1
#                 return self.__obj[row]
#             raise StopIteration

    class AccessProxy:
        def __init__(self, obj, row):
            self.__obj = obj
            self.__row = row
        
        @property
        def dims(self):
            return self.__obj.__dims
        
        @property
        def shape(self):
            return self.__obj.__shape
        
        def __str__(self):
            row = self.__row
            rows, columns = self.__obj.__shape
            values = ['{}'.format(v) for v in self.__obj.__data[row*columns:(row+1)*columns]]
            return '[{}]\n'.format(',\t'.join(values))
        
        @micropython.native
        def __iter__(self):
            return Matrix.ColumnIterator(self.__obj, self.__row)
    
        @micropython.native        
        def __getitem__(self, column):
            rows, columns = self.__obj.shape
            idx = self.__row * columns + column    
            return self.__obj.__data[idx]
        
        @micropython.native        
        def __setitem__(self, column, value):
            rows, columns = self.__obj.shape
            idx = self.__row * columns + column    
            self.__obj.__data[idx] = value
            
        @micropython.native
        def row(self):
            m = self.__obj.__class__(dims = self.__obj.__dims)
            columns = self.__obj.__shape[1]
            dest_addr_start, dest_addr_end = self.__row*columns, (self.__row + 1)*columns            
            m.append_row(*self.__obj.__data[dest_addr_start:dest_addr_end])            
            return m

        @micropython.native
        def __add__(self, other):
            m = self.__obj.__class__(dims = self.__obj.__dims)
            columns = self.__obj.__shape[1]
            if isinstance(other, Matrix):
                if (not other.shape[1] == self.__obj.__shape[1]) or (not other.shape[0] == 1):
                    raise Exception('Invalid shapes for row add proxy({}) and matrix({}, {})'.format(self.__shape[1], *other.__shape))
                m.__shape = (1, columns)
                dest_addr_start, dest_addr_end = self.__row*columns, (self.__row + 1)*columns
                m.append_row(*[p + o for p, o in zip(self.__obj.__data[dest_addr_start:dest_addr_end], self.__data[0:columns])])
            elif isinstance(other, Matrix.AccessProxy):
                if not (other.shape[1] == columns):
                    raise Exception('Invalid shapes for row add proxy({}) and proxy({})'.format(self.__shape[1], other.shape[1]))
                m.__shape = (1, columns)
                dest_addr_start, dest_addr_end = self.__row*columns, (self.__row + 1)*columns
                src_addr_start, src_addr_end = other.__row*columns, (other.__row + 1)*columns
                m.append_row(*[p + o for p, o in zip(self.__obj.__data[dest_addr_start:dest_addr_end], self.__obj.__data[src_addr_start:src_addr_end])])
            else:
                dest_addr_start, dest_addr_end = self.__row*columns, (self.__row + 1)*columns
                m.append_row(*[other + v for v in self.__obj.__data[dest_addr_start:dest_addr_end]])
            return m

        @micropython.native
        def __radd__(self, other):
            return self.__add__(other)
        
        @micropython.native
        def __iadd__(self, other):
            columns = self.__obj.__shape[1]
            if isinstance(other, Matrix):
                if (not other.shape[1] == self.__obj.__shape[1]) or (not other.shape[0] == 1):
                    raise Exception('Invalid shapes for row add proxy({}) and matrix({}, {})'.format(self.__shape[1], *other.__shape))
                dest_base = self.__row*columns
                for offset in range(columns):
                    self.__obj.__data[dest_base + offset] += other.__data[offset]
            elif isinstance(other, Matrix.AccessProxy):
                if not (other.shape[1] == columns):
                    raise Exception('Invalid shapes for row add proxy({}) and proxy({})'.format(self.__shape[1], other.shape[1]))
                dest_base = self.__row*columns
                src_base = other.__row*columns
                for offset in range(columns):
                    self.__obj.__data[dest_base + offset] += other.__obj.__data[src_base + offset]
            else:
                dest_base = self.__row*columns
                for offset in range(columns):
                    self.__obj.__data[dest_base + offset] += other
            return self
        
        @micropython.native
        def __sub__(self, other):
            m = self.__obj.__class__(dims = self.__obj.__dims)
            columns = self.__obj.__shape[1]
            if isinstance(other, Matrix):
                if (not other.shape[1] == self.__obj.__shape[1]) or (not other.shape[0] == 1):
                    raise Exception('Invalid shapes for row sub proxy({}) and matrix({}, {})'.format(self.__shape[1], *other.__shape))
                m.__shape = (1, columns)
                dest_addr_start, dest_addr_end = self.__row*columns, (self.__row + 1)*columns
                m.append_row(*[o - p for p, o in zip(self.__obj.__data[dest_addr_start:dest_addr_end], other.__data[0:columns])])
            elif isinstance(other, Matrix.AccessProxy):
                if not (other.shape[1] == columns):
                    raise Exception('Invalid shapes for row sub proxy({}) and proxy({})'.format(self.__shape[1], other.shape[1]))
                m.__shape = (1, columns)
                dest_addr_start, dest_addr_end = self.__row*columns, (self.__row + 1)*columns
                src_addr_start, src_addr_end = other.__row*columns, (other.__row + 1)*columns
                m.append_row(*[o - p for p, o in zip(self.__obj.__data[dest_addr_start:dest_addr_end], other.__obj.__data[src_addr_start:src_addr_end])])
            else:
                dest_addr_start, dest_addr_end = self.__row*columns, (self.__row + 1)*columns
                m.append_row(*[other - v for v in self.__obj.__data[dest_addr_start:dest_addr_end]])
            return m

        @micropython.native
        def __rsub__(self, other):
            m = self.__obj.__class__(dims = self.__obj.__dims)
            columns = self.__obj.__shape[1]
            if isinstance(other, Matrix):
                if (not other.shape[1] == columns) or (not other.shape[0] == 1):
                    raise Exception('Invalid shapes for row sub proxy({}) and matrix({}, {})'.format(columns, *other.__shape))
                m.__shape = (1, columns)
                dest_addr_start, dest_addr_end = self.__row*columns, (self.__row + 1)*columns
                m.append_row(*[o - p for p, o in zip(self.__obj.__data[dest_addr_start:dest_addr_end], self.__data[0:columns])])
            elif isinstance(other, Matrix.AccessProxy):
                if not (other.shape[1] == columns):
                    raise Exception('Invalid shapes for row sub proxy({}) and proxy({})'.format(columns, other.shape[1]))
                m.__shape = (1, columns)
                dest_addr_start, dest_addr_end = self.__row*columns, (self.__row + 1)*columns
                src_addr_start, src_addr_end = other.__row*columns, (other.__row + 1)*columns
                m.append_row(*[o - p for p, o in zip(self.__obj.__data[dest_addr_start:dest_addr_end], self.__obj.__data[src_addr_start:src_addr_end])])
            else:
                dest_addr_start, dest_addr_end = self.__row*columns, (self.__row + 1)*columns
                m.append_row(*[v - other for v in self.__obj.__data[dest_addr_start:dest_addr_end]])
            return m
        
        @micropython.native
        def __isub__(self, other):
            columns = self.__obj.__shape[1]
            if isinstance(other, Matrix):
                if (not other.shape[1] == self.__obj.__shape[1]) or (not other.shape[0] == 1):
                    raise Exception('Invalid shapes for row sub proxy({}) and matrix({}, {})'.format(self.__shape[1], *other.__shape))
                dest_base = self.__row*columns
                for offset in range(columns):
                    self.__obj.__data[dest_base + offset] -= other.__data[offset]
            elif isinstance(other, Matrix.AccessProxy):
                if not (other.shape[1] == columns):
                    raise Exception('Invalid shapes for row sub proxy({}) and proxy({})'.format(self.__shape[1], other.shape[1]))
                dest_base = self.__row*columns
                src_base = other.__row*columns
                for offset in range(columns):
                    self.__obj.__data[dest_base + offset] -= other.__obj.__data[src_base + offset]
            else:
                dest_base = self.__row*columns
                for offset in range(columns):
                    self.__obj.__data[dest_base + offset] -= other
            return self

        @micropython.native
        def __mul__(self, other):
            m = self.__obj.__class__(dims = self.__obj.__dims)
            columns = self.__obj.__shape[1]
            dest_addr_start, dest_addr_end = self.__row*columns, (self.__row + 1)*columns
            m.append_row(*[other * v for v in self.__obj.__data[dest_addr_start:dest_addr_end]])
            return m

        @micropython.native
        def __rmul__(self, other):
            return self.__mul__(other)

        @micropython.native
        def __imul__(self, other):
            columns = self.__obj.__shape[1]
            dest_base = self.__row*columns
            for offset in range(columns):
                self.__obj.__data[dest_base + offset] *= other
            return self

#     class ColumnIterator:
#         def __init__(self, obj, row):
#             self.__obj = obj
#             self.__column = 0
#             columns = self.__obj.__shape[1]
#             self.__columns = columns
#             self.__base = row * columns
#             self.__data = self.__obj.__data
# 
#         @micropython.native
#         def __next__(self):
#             column = self.__column
#             rows, columns = self.__obj.__shape
#             if column < columns:
#                 self.__column += 1
#                 idx = self.__base + column
#                 return self.__data[idx]
#             raise StopIteration
        
    @micropython.native
    def __init__(self, data=None, dims=2, shape=None, raw_data=False):
        self.__dims = dims
        self.__shape = shape
        self.__data = []
        if (data is not None):
            if raw_data:
                self.__data = data
            else:
                self.__from_data(data)        
 
    @property
    def dims(self):
        return self.__dims
    
    @property
    def shape(self):
        return self.__shape
    
    @micropython.native
    def set_shape(self, shape):
        self.__shape = shape
        return self.__shape
    
    @property
    def data(self):
        return self.__data
    
    @micropython.native
    def set_data(self, data):
        self.__data = data
        return self.__data
    
    @micropython.native
    def __str__(self):
        text = ''
        try:
            rows, columns = self.shape
            for row in range(rows):
                values = ['{}'.format(v) for v in self.__data[row*columns:(row+1)*columns]]
                text += '[{}]\n'.format(',\t'.join(values))
        except:
            if (self.__shape is None) or (self.__data is None):
                text = 'Matrix()'
        return text

    @micropython.native
    def __iter__(self):
        return Matrix.RowIterator(self)

    @micropython.native        
    def __getitem__(self, row):
        return Matrix.AccessProxy(self, row)

    @micropython.native        
    def append_row(self, *row_data):
        shape = self.__shape
        if shape is None:
            shape = (1, len(row_data))
        else:
            rows, columns = shape
            if not columns == len(row_data):
                raise Exception('Attempting to append a row of {} values in a matrix of {} columns'.format(len(row_data), columns))                
            shape = (rows + 1, columns)
        self.__data.extend(row_data)            
        self.__shape = shape

    @micropython.native
    def transpose(self):
        m = self.__class__()
        rows, columns = self.__shape
        src_data = self.__data
        if (rows > 1) and (columns > 1):
            new_data = []
            for row in range(columns):
                new_data.extend(src_data[row::columns])
            m.__data = new_data 
        else:  # Optimized for Vectors
            m.__data = src_data[:]
        m.__shape = (columns, rows)
        return m

    @micropython.native
    def copy(self):
        m = self.__class__(dims = self.__dims)
        m.__data = self.__data[:]
        m.__shape = self.__shape
        return m
        
    @micropython.native
    def __neg__(self):
        m = self.__class__(dims = self.__dims)
        m.__data = [-d for d in self.__data]
        m.__shape = self.__shape
        return m

    @micropython.native
    def __add__(self, other):
        m = self.__class__(dims = self.__dims)
        m.__shape = self.__shape

        if isinstance(other, Matrix):
            if not (self.__shape == other.__shape):
                raise Exception('Invalid matrix sizes for add with {} and {}'.format(self.__shape, other.__shape))
            m.__data = [(s+o) for s, o in zip(self.__data, other.__data)]
        else:
            m.__data = [(s+other) for s in self.__data]
        return m

    @micropython.native
    def __radd__(self, v):
        return self.__add__(v)
    
    @micropython.native
    def __iadd__(self, v):
        self.__data = [(s+v) for s in self.__data]        
        return self
        
    @micropython.native
    def __sub__(self, other):
        m = self.__class__(dims = self.__dims)
        m.__shape = self.__shape

        if isinstance(other, Matrix):
            if not (self.__shape == other.__shape):
                raise Exception('Invalid matrix sizes for add with {} and {}'.format(self.__shape, other.__shape))
            m.__data = [(s-o) for s, o in zip(self.__data, other.__data)]
        else:
            m.__data = [(s-other) for s in self.__data]
        return m

    @micropython.native
    def __rsub__(self, other):
        m = self.__class__(dims = self.__dims)
        m.__shape = self.__shape
        m.__data = [(other-s) for s in self.__data]
        return m
    
    @micropython.native
    def __isub__(self, v):
        self.__data = [(s-v) for s in self.__data]        
        return self
    
    @micropython.native
    def __mul__(self, other):
        if isinstance(other, Matrix):
            if not (self.__shape[1] == other.__shape[0]):
                raise Exception('Invalid matrix sizes for multiplication {} and {}'.format(self.__shape, other.__shape))
            l_data = self.__data
            r_data = other.__data
            m = self.__class__()            
            (rows, src_columns), columns = self.__shape, other.__shape[1]
            data = []
            for row in range(rows):
                for column in range(columns):
                    dot = 0
                    l_data_idx_base = row*src_columns
                    r_data_idx = column
                    for src_column in range(src_columns):
                        dot += (l_data[l_data_idx_base + src_column] * r_data[r_data_idx])
                        r_data_idx += columns
                    data.append(dot)
            m.__data = data
            m.__shape = (rows, columns)
            return data[0] if (rows, columns) == (1, 1) else m        
        else:
            return self.__rmul__(other)            
        return None
    
    @micropython.native
    def __rmul__(self, other):
        m = self.__class__(dims = self.__dims)
        m.__shape = self.__shape
        m.__data = [(s*other) for s in self.__data]
        return m
    
    @micropython.native
    def __imul__(self, v):
        self.__data = [(v*s) for s in self.__data]        
        return self
    
    @micropython.native
    def __from_data(self, data):
        shape = []
        shape_initialized = False
        for d in data: # TODO: THIS ASSUMES A 2D MATRIX
            if not shape_initialized:
                shape_initialized = True                
                shape.append(1)                
                if isinstance(d, list) or isinstance(d, tuple):
                    shape.append(len(d))
                else:
                    shape.append(len(data))
                    self.append_row(*data)
                    break
            else:
                if not len(d) == shape[1]:
                    raise Exception # TODO: THIS SHOULD BE A MORE SPECIALIZED EXCEPTION
                shape[0] += 1
            self.__data.extend(d)
        self.__shape = tuple(shape)

       
def dot(a, b):
    a_min, a_max = min(*a.shape), max(*b.shape)
    b_min, b_max = min(*a.shape), max(*b.shape)
    
    if not ((a_min == 1) and (b_min == 1) and (a_max == b_max)):
        raise Exception('Invalid matrix sizes for dot product {} and {}'.format(self.__shape, other.__shape))

    if (a.shape[0] == 1) and (not b.shape[0] == 1):
        return a * b
    elif (not a.shape[0] ==1) and (b.shape[0] == 1):
        return b * a
    return a * b.transpose()
      
     
# @micropython.native
# def backsub(A, b):
#     rows = A.shape[0]
#     b_columns = b.shape[1]
#     for idx in reversed(range(rows)):
#         head_val = A[idx][idx]
#         A[idx][idx] = 1
#         for row in range(idx):
#             scale = A[row][idx] / head_val
#             A[row][idx] = 0
#             for column in range(b_columns):
#                 b[row][column] -= (scale * b[idx][column])
#         for column in range(b_columns):
#             b[idx][column] /= head_val
#     return b

@micropython.native
def backsub(A, b):
    rows, A_columns = A.shape
    b_columns = b.shape[1]
    A_data = A.data
    b_data = b.data
    for idx in reversed(range(rows)):
        head_idx = idx*A_columns + idx
        head_val = A_data[head_idx]
        b_idx_base = idx*b_columns
        for row in range(idx):
            pivot_idx = row*A_columns + idx
            scale = A_data[pivot_idx] / head_val
            b_row_base = row*b_columns
            for column in range(b_columns):
                b_data[b_row_base + column] -= (scale * b_data[b_idx_base + column])
        for column in range(b_columns):
            b_data[b_idx_base + column] /= head_val
    b.set_data(b_data)
    return b


# @micropython.native
# def solve(A, b, method='gaussian-elimination'):
#     rows, columns = A.shape
#     b_columns = b.shape[1]
#     for head in range(A.shape[0]):
#         max_head = head
#         
#         # Do Partial Pivot
#         max_pivot = abs(A[head][head])
#         for row in range(head + 1, rows):
#             val = A[row][head]
#             if abs(val) > max_pivot:
#                 max_pivot = val
#                 max_head = row
#         if max_head > head:
#             for column in range(head, rows):
#                 temp = A[head][column]
#                 A[head][column] = A[max_head][column]
#                 A[max_head][column] = temp
#             for column in range(b_columns):
#                 temp = b[head][column]
#                 b[head][column] = b[max_head][column]
#                 b[max_head][column] = temp
#                 
#         # Cancel Lower-Left Submatrix
#         v = A[head][head]
#         A[head][head] = 1
#         
#         for row in range(head + 1, rows):
#             scale = A[row][head] / v
#             A[row][head] = 0
#             for column in range(head+1, columns):
#                 A[row][column] -= (scale*A[head][column])
#             for column in range(b_columns):
#                 b[row][column] -= (scale*b[head][column])
#                 
#         # Correct the Reference Line
#         for column in range(head+1, columns):
#             A[head][column] /= v
#         for column in range(b_columns):
#             b[head][column] /= v
# 
#     return backsub(A, b)


@micropython.native
def solve(A, b, method='gaussian-elimination'):
    rows, columns = A.shape
    b_columns = b.shape[1]
    A_data = A.data
    b_data = b.data
    for head in range(rows):
        max_head = head
        
        # Do Partial Pivot
        head_idx = head * columns + head
        max_pivot = abs(A_data[head * columns + head])
        for row in range(head + 1, rows):
            val = A_data[row * columns + head]
            if abs(val) > max_pivot:
                max_pivot = val
                max_head = row
        if max_head > head:
            head_base = head * columns
            max_head_base = max_head * columns
            for column in range(head, rows):
                temp_head_idx = head_base + column
                temp_max_head_idx = max_head_base + column
                temp = A_data[temp_head_idx]
                A_data[temp_head_idx] = A_data[temp_max_head_idx]
                A_data[temp_max_head_idx] = temp
            head_base = head * b_columns
            max_head_base = max_head * b_columns
            for column in range(b_columns):
                temp_head_idx = head_base + column
                temp_max_head_idx = max_head_base + column
                temp = b_data[temp_head_idx]
                b_data[temp_head_idx] = b_data[temp_max_head_idx]
                b_data[temp_max_head_idx] = temp
                
        # Cancel Lower-Left Submatrix
        v = A_data[head_idx]
        A_data[head_idx] = 1
        
        A_head_base = head * columns
        b_head_base = head * b_columns
        for row in range(head + 1, rows):
            row_head_idx = row * columns + head
            scale = A_data[row_head_idx] / v
            A_data[row_head_idx] = 0
            row_base = row * columns
            for column in range(head+1, columns):
                A_data[row_base + column] -= (scale*A_data[A_head_base + column])
            row_base = row * b_columns
            for column in range(b_columns):
                b_data[row_base + column] -= (scale*b_data[b_head_base + column])
                
        # Correct the Reference Line
        for column in range(head+1, columns):
            A_data[A_head_base + column] /= v
        for column in range(b_columns):
            b_data[b_head_base + column] /= v
    A.set_data(A_data)
    b.set_data(b_data)
    return backsub(A, b)


@micropython.native
def inv(A):
    return solve(A, Matrix.Eye(A.shape[0]))


# @micropython.native
# def ols(X, y):
#     X_t = X.transpose()
#     X_tX = X_t * X
#     X_ty = X_t * y
#     return solve(X_tX, X_ty)


def octaveFormatMatrix(m):
    rows, columns = m.shape
    text = '['
    for row in range(rows):
        for column in range(columns):
            if column > 0:
                text += ' '
            text += '{}'.format(m[row][column])
        if row < (rows-1):
            text += ';'
    text += ']'
    return text
    
@micropython.native
def ols(X, y):
#     print('X = {}'.format(octaveFormatMatrix(X)))
#     print('y = {}'.format(octaveFormatMatrix(y)))
    X_rows, X_columns = X.shape
    y_rows, y_columns = y.shape
    X_data = X.data
    y_data = y.data
    X_tX_data = []
    X_ty_data = []
    for row in range(X_columns):
        for column in range(X_columns):
            X_dot = 0
            r_idx = column
            for idx in range(X_rows):
                X_dot += (X_data[idx * X_columns + row] * X_data[r_idx])
                r_idx += X_columns
            X_tX_data.append(X_dot)
        for column in range(y_columns):
            y_dot = 0
            r_idx = column
            for idx in range(X_rows):
                y_dot += (X_data[idx * X_columns + row] * y_data[r_idx])
                r_idx += y_columns
            X_ty_data.append(y_dot)
    X_tX = Matrix(X_tX_data, shape=(X_columns, X_columns), raw_data=True)
    X_ty = Matrix(X_ty_data, shape=(X_columns, y_columns), raw_data=True)
    b = solve(X_tX, X_ty)
    del X_tX
    del X_ty
    del X_tX_data
    del X_ty_data
    return b


if __name__ == '__main__':
    import gc
    X = Matrix()
    del X
    gc.collect()
    