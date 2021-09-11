package main

type (
	Matrix struct {
		Data []float32
		Rows uint32
		Cols uint32
	}

	Transposed struct {
		Matrix *Matrix
	}

	MatrixLike interface {
		GetCols() uint32
		GetRows() uint32
		Get(i, j uint32) float32
	}
)

func SingletonMatrix(rows uint32, data []float32) *Matrix {
	return NewMatrix(rows, 1, data)
}

func NewMatrix(rows, cols uint32, data []float32) *Matrix {
	if len(data) != int(rows*cols) {
		panic("Wrong matrix dimensions")
	}

	return &Matrix{
		Data: data,
		Rows: rows,
		Cols: cols,
	}
}

func EmptyMatrix(rows, cols uint32) *Matrix {
	return NewMatrix(rows, cols, make([]float32, rows*cols))
}

func FromTemplate(mat *Matrix) *Matrix {
	return NewMatrix(mat.Rows, mat.Cols, make([]float32, mat.Size()))
}

func (m *Matrix) Size() uint32 {
	return m.Rows * m.Cols
}

func (m *Matrix) Get(row, col uint32) float32 {
	if row >= m.Rows || col >= m.Cols {
		panic("Bad Address")
	}
	return m.Data[col*m.Rows+row]
}

func (m *Matrix) Set(row, col uint32, v float32) {
	if row >= m.Rows || col >= m.Cols {
		panic("Bad Address")
	}
	m.Data[col*m.Rows+row] = v
}

func (m *Matrix) AddTo(row, col uint32, v float32) {
	if row >= m.Rows || col >= m.Cols {
		panic("Bad Address")
	}
	m.Data[col*m.Rows+row] += v
}

func (m *Matrix) T() *Transposed {
	return &Transposed{Matrix: m}
}

func (m *Matrix) Dot(fst, snd MatrixLike) {
	if m.Cols != snd.GetCols() || m.Rows != fst.GetCols() || fst.GetCols() != snd.GetCols() {
		panic("Incompatible matrices for multiplication")
	}
	for i := uint32(0); i < fst.GetRows(); i++ {
		for j := uint32(0); j < snd.GetCols(); j++ {
			m.Set(j, i, 0)
			for k := uint32(0); k < fst.GetCols(); k++ {
				m.AddTo(j, i, fst.Get(i, k)*snd.Get(k, j))
			}
		}
	}
}

func (m *Matrix) GetRows() uint32 {
	return m.Rows
}

func (m *Matrix) GetCols() uint32 {
	return m.Cols
}

func (m *Transposed) GetRows() uint32 {
	return m.Matrix.Cols
}

func (m *Transposed) GetCols() uint32 {
	return m.Matrix.Rows
}

func (m *Transposed) Get(row, col uint32) float32 {
	return m.Matrix.Data[row*m.GetCols()+col]
}

// func (m *Matrix) Mean() float32 {
// 	if m.Size() == 1 {
// 		return m.Data[0]
// 	}
//
// 	acc := float32(0)
// 	for i := uint32(0); i < m.Size(); i++ {
// 		acc += m.Data[i]
// 	}
//
// 	return acc / float32(m.Size())
// }

func (m *Matrix) SumColumns(mat *Matrix) {
	if m.Cols != mat.Cols {
		panic("Cannot fit mat in m")
	}
	for i := uint32(0); i < mat.Cols; i++ {
		m.Data[i] = 0
		for j := uint32(0); j < mat.Rows; j++ {
			m.Data[i] += m.Get(j, i)
		}
	}
}

func (m *Matrix) Scale(mat *Matrix, scaleFactor float32) {
	for i := uint32(0); i < m.Size(); i++ {
		m.Data[i] = mat.Data[i] * scaleFactor
	}
}

func (m *Matrix) Multiply(fst *Matrix, snd *Matrix) {
	if m.Cols != fst.Cols || m.Cols != snd.Cols ||
		m.Rows != fst.Rows || m.Rows != snd.Rows {
		panic("Bad sized matrices")
	}
	for i := uint32(0); i < m.Size(); i++ {
		m.Data[i] = fst.Data[i] * snd.Data[i]
	}
}

func (m *Matrix) Apply(mat *Matrix, fn func(float32) float32) {
	for i := uint32(0); i < m.Size(); i++ {
		m.Data[i] = fn(mat.Data[i])
	}
}

func (m *Matrix) AddAndApply(fst, snd *Matrix, fn func(float32) float32) {
	if m.Cols != fst.Cols || m.Cols != snd.Cols ||
		m.Rows != fst.Rows || m.Rows != snd.Rows {
		panic("Bad sized matrices")
	}
	for i := uint32(0); i < m.Size(); i++ {
		m.Data[i] = fn(fst.Data[i] + snd.Data[i])
	}
}

func (m *Matrix) Add(fst, snd *Matrix) {
	identity := func(x float32) float32 {
		return x
	}

	m.AddAndApply(fst, snd, identity)
}

func (m *Matrix) Subtract(fst, snd *Matrix) {
	if m.Cols != fst.Cols || m.Cols != snd.Cols ||
		m.Rows != fst.Rows || m.Rows != snd.Rows {
		panic("Bad sized matrices")
	}
	for i := uint32(0); i < m.Size(); i++ {
		m.Data[i] = fst.Data[i] - snd.Data[i]
	}
}

func (output *Matrix) ForwardPropagate(input, weights, biases *Matrix, activation func(float32) float32) {
	output.Dot(input, weights)
	output.AddAndApply(output, biases, activation)
}
