package main

type (
	Matrix struct {
		Data []float32
		Rows uint32
		Cols uint32
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

func (m *Matrix) Size() uint32 {
	return m.Rows * m.Cols
}

func (m *Matrix) Get(row, col uint32) float32 {
	return m.Data[col*m.Rows+row]
}

func (m *Matrix) Set(row, col uint32, v float32) {
	m.Data[col*m.Rows+row] = v
}

func (m *Matrix) AddTo(row, col uint32, v float32) {
	m.Data[col*m.Rows+row] += v
}

func (m *Matrix) Product(fst, snd *Matrix) {
	for i := uint32(0); i < fst.Rows; i++ {
		for j := uint32(0); j < snd.Cols; j++ {
			m.Set(j, i, 0)
			for k := uint32(0); k < fst.Cols; k++ {
				m.AddTo(j, i, fst.Get(i, k)*snd.Get(k, j))
			}
		}
	}
}

func (m *Matrix) Add(other *Matrix, fn func(float32) float32) {
	for i := uint32(0); i < m.Rows; i++ {
		for j := uint32(0); j < m.Cols; j++ {
			m.Set(j, i, fn(m.Get(i, j)+other.Get(i, j)))
		}
	}
}

func (m *Matrix) Subtract(other *Matrix) {
	for i := uint32(0); i < m.Rows; i++ {
		for j := uint32(0); j < m.Cols; j++ {
			m.Set(j, i, m.Get(i, j)-other.Get(i, j))
		}
	}
}

func (output *Matrix) ForwardPropagate(input, weights, biases *Matrix, activation func(float32) float32) {
	output.Product(input, weights)
	output.Add(biases, activation)
}
