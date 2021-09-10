package main

type (
	Matrix struct {
		Data []float32
		Rows int
		Cols int
	}
)

func (m *Matrix) Size() int {
	return m.Rows * m.Cols
}

func (m *Matrix) Get(row, col int) float32 {
	return m.Data[col*m.Rows+row]
}

func (m *Matrix) Set(row, col int, v float32) {
	m.Data[col*m.Rows+row] = v
}

func (m *Matrix) Add(row, col int, v float32) {
	m.Data[col*m.Rows+row] += v
}

func (output *Matrix) ForwardPropagate(input Matrix,
	weights Matrix,
	biases Matrix,
	activation func(float32) float32) {

	for i := 0; i < input.Rows; i++ {
		for j := 0; j < weights.Cols; j++ {
			output.Set(j, i, 0)

			for k := 0; k < input.Cols; k++ {
				output.Add(j, i, input.Get(i, k)*weights.Get(k, j))
			}
			output.Set(i, j, activation(output.Get(i, j)+biases.Get(i, j)))
		}
	}
}
