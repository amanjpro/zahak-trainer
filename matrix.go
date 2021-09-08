package main

import "fmt"

func main() {
	fmt.Println("vim-go")
}

type (
	Matrix struct {
		Rows    int
		Columns int
		Data    []float64
	}
)

func NewMatrix(rows, colums int) *Matrix {
	return &Matrix{
		Rows:    rows,
		Columns: columns,
		Data:    make([]float64, rows*columns),
	}
}

func (m *Matrix) Get(row, column int) float64 {
	return m.Data[column*m.Rows+row]
}

func (m *Matrix) ElementAt(index int) float64 {
	return m.Data[index]
}

func (m *Matrix) Fill(value float64) {
	for i := 0; i < len(m.Data); i++ {
		m.Data[i] = value
	}
}
