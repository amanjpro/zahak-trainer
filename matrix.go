package main

import "fmt"

type (
	Matrix struct {
		Data []float32
		Rows uint32
		Cols uint32
	}
)

func SingletonMatrix(rows uint32, data []float32) Matrix {
	return NewMatrix(rows, 1, data)
}

func NewMatrix(rows, cols uint32, data []float32) Matrix {
	if len(data) != int(rows*cols) {
		panic("Wrong matrix dimensions")
	}

	return Matrix{
		Data: data,
		Rows: rows,
		Cols: cols,
	}
}

func (m *Matrix) Size() uint32 {
	return m.Rows * m.Cols
}

func (m *Matrix) Get(row, col uint32) float32 {
	if row >= m.Rows || col >= m.Cols {
		fmt.Println("Bad Address", col, m.Cols, row, m.Rows)
		panic("Bad Address")
	}
	return m.Data[col*m.Rows+row]
}

func (m *Matrix) Reset() {
	for i := uint32(0); i < m.Size(); i++ {
		m.Data[i] = 0
	}
}
