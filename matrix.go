package main

import "fmt"

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

func EmptyMatrix(rows, cols uint32) Matrix {
	return NewMatrix(rows, cols, make([]float32, rows*cols))
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

func (m *Matrix) Set(row, col uint32, v float32) {
	if row >= m.Rows || col >= m.Cols {
		fmt.Println(m.Rows, row, m.Cols, col)
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
	// - The number of columns of the 1st matrix must equal the number of rows of the 2nd matrix.
	// - And the result will have the same number of rows as the 1st matrix, and the same number of columns as the 2nd matrix.
	if m.Cols != snd.GetCols() || m.Rows != fst.GetRows() || fst.GetCols() != snd.GetRows() {
		fmt.Println(m.Cols != snd.GetCols(), m.Rows != fst.GetRows(), fst.GetCols() != snd.GetRows())
		fmt.Println(m.Cols, snd.GetCols(), m.Rows, fst.GetRows(), fst.GetCols(), snd.GetRows())
		fmt.Println(m.Rows, m.Cols, fst.GetRows(), fst.GetCols(), snd.GetRows(), snd.GetCols())
		panic("Incompatible matrices for multiplication")
	}
	frows := fst.GetRows()
	fcols := fst.GetCols()
	scols := snd.GetCols()
	for i := uint32(0); i < frows; i++ {
		for j := uint32(0); j < scols; j++ {
			m.Set(i, j, 0)
			for k := uint32(0); k < fcols; k++ {
				m.AddTo(i, j, fst.Get(i, k)*snd.Get(k, j))
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

func (m *Matrix) SumColumns(mat Matrix) {
	if m.Cols != mat.Cols {
		panic("Cannot fit mat in m")
	}
	for i := uint32(0); i < mat.Cols; i++ {
		m.Data[i] = 0
		for j := uint32(0); j < mat.Rows; j++ {
			m.Data[i] += mat.Get(j, i)
		}
	}
}

func (m *Matrix) Scale(mat Matrix, scaleFactor float32) {
	m.Apply(mat, func(x float32) float32 { return x * scaleFactor })
}

func (m *Matrix) Multiply(fst Matrix, snd Matrix) {
	if m.Cols != fst.Cols || m.Cols != snd.Cols ||
		m.Rows != fst.Rows || m.Rows != snd.Rows {
		fmt.Println(m.Rows, m.Cols, fst.GetRows(), fst.GetCols(), snd.GetRows(), snd.GetCols())
		panic("Bad sized matrices")
	}
	for i := uint32(0); i < m.Size(); i++ {
		m.Data[i] = fst.Data[i] * snd.Data[i]
	}
}

func (m *Matrix) Apply(mat Matrix, fn func(float32) float32) {
	for i := uint32(0); i < m.Size(); i++ {
		m.Data[i] = fn(mat.Data[i])
	}
}

func (m *Matrix) AddAndApply(fst, snd Matrix, fn func(float32) float32) {
	if m.Cols != fst.Cols || m.Cols != snd.Cols ||
		m.Rows != fst.Rows || m.Rows != snd.Rows {
		fmt.Println(m.Rows, m.Cols, fst.GetRows(), fst.GetCols(), snd.GetRows(), snd.GetCols())
		panic("Bad sized matrices")
	}
	for i := uint32(0); i < m.Size(); i++ {
		m.Data[i] = fn(fst.Data[i] + snd.Data[i])
	}
}

func (m *Matrix) Add(fst, snd Matrix) {
	identity := func(x float32) float32 {
		return x
	}

	m.AddAndApply(fst, snd, identity)
}

func (m *Matrix) Subtract(fst, snd MatrixLike) {
	if m.Cols != fst.GetCols() || m.Cols != snd.GetCols() ||
		m.Rows != fst.GetRows() || m.Rows != snd.GetRows() {
		fmt.Println(m.Rows, m.Cols, fst.GetRows(), fst.GetCols(), snd.GetRows(), snd.GetCols())
		panic("Bad sized matrices")
	}
	cols := fst.GetCols()
	rows := fst.GetRows()
	for i := uint32(0); i < cols; i++ {
		for j := uint32(0); i < rows; i++ {
			m.Data[i] = fst.Get(i, j) - snd.Get(i, j)
		}
	}
}

func Multiply(fst, snd Matrix) Matrix {
	m := EmptyMatrix(fst.Rows, fst.Cols)
	m.Multiply(fst, snd)
	return m
}

func Add(fst, snd Matrix) Matrix {
	m := EmptyMatrix(fst.Rows, fst.Cols)
	m.Add(fst, snd)
	return m
}

func Subtract(fst, snd Matrix) Matrix {
	m := EmptyMatrix(fst.Rows, fst.Cols)
	m.Subtract(&fst, &snd)
	return m
}

func Apply(fst Matrix, fn func(x float32) float32) Matrix {
	m := EmptyMatrix(fst.Rows, fst.Cols)
	m.Apply(fst, fn)
	return m
}

func AddAndApply(fst, snd Matrix, fn func(x float32) float32) Matrix {
	m := EmptyMatrix(fst.Rows, fst.Cols)
	m.AddAndApply(fst, snd, fn)
	return m
}

func Dot(fst, snd MatrixLike) Matrix {
	m := EmptyMatrix(fst.GetRows(), snd.GetCols())
	m.Dot(fst, snd)
	return m
}

func SumColumns(mat Matrix) Matrix {
	m := EmptyMatrix(1, mat.GetCols())
	m.SumColumns(mat)
	return m
}
func Scale(mat Matrix, scaleFactor float32) Matrix {
	return Apply(mat, func(x float32) float32 { return x * scaleFactor })
}
