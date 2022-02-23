class Matrix:
    def __init__(self, data) -> None:
        self.data = data

    @property
    def shape(self):
        return (len(self.data), len(self.data[0]))

    def __getitem__(self, index):
        return self.data[index]

    def is_elementwise_compatible(self, other):
        assert (
            self.shape == other.shape
        ), f"operands could not be broadcast together with shapes {self.shape} {other.shape}"

    def is_matrixwise_compatible(self, other):
        assert (
            self.shape[1] == other.shape[0]
        ), f"operands could not be broadcast together with shapes {self.shape} {other.shape}"

    def __add__(self, other):
        self.is_elementwise_compatible(other)
        rows_len, cols_len = self.shape

        result = []

        for i in range(rows_len):
            tmp = []
            for j in range(cols_len):
                tmp.append(self[i][j] + other[i][j])
            result.append(tmp)

        return Matrix(result)

    def __mul__(self, other):
        self.is_elementwise_compatible(other)
        rows_len, cols_len = self.shape

        result = []

        for i in range(rows_len):
            tmp = []
            for j in range(cols_len):
                tmp.append(self[i][j] * other[i][j])
            result.append(tmp)

        return Matrix(result)

    def _get_column(self, index):
        return [row[index] for row in self]

    @staticmethod
    def _multiply_row_by_col(row, col):
        result = 0
        for i in range(len(row)):
            result += row[i] * col[i]

        return result

    def __matmul__(self, other):
        self.is_matrixwise_compatible(other)

        rows_len, cols_len = self.shape[0], other.shape[1]

        result = []
        for i in range(rows_len):
            tmp = []
            for j in range(cols_len):
                tmp.append(self._multiply_row_by_col(self[i], other._get_column(j)))
            result.append(tmp)

        return Matrix(result)

    def __str__(self):
        return "\n".join(["\t".join([str(cell) for cell in row]) for row in self])
