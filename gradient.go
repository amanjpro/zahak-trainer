package main

import (
	"fmt"
	"math"
)

type (
	Gradient struct {
		Value float32
		M1    float32
		M2    float32
	}

	Gradients struct {
		Data []Gradient
		Rows uint32
		Cols uint32
	}
)

const (
	Beta1 float32 = 0.9
	Beta2 float32 = 0.999
)

// Implementing Gradient

func (g *Gradient) Update(delta float32) {
	g.Value += delta
}

func (g *Gradient) Calculate() float32 {

	if g.Value == 0 {
		// nothing to calculate
		return 0
	}

	g.M1 = g.M1*Beta1 + g.Value*(1-Beta1)
	g.M2 = g.M2*Beta2 + (g.Value*g.Value)*(1-Beta2)

	return LearningRate * g.M1 / float32(math.Sqrt(float64(g.M2))+1e-8)
}

func (g *Gradient) Reset() {
	g.Value = 0.0
}

func (g *Gradient) Apply(elem *float32) {
	*elem -= g.Calculate()
	g.Reset()
}

// Implementing Gradients (a matrix of Gradient)

func NewGradients(rows, cols uint32) Gradients {

	return Gradients{
		Data: make([]Gradient, cols*rows),
		Rows: rows,
		Cols: cols,
	}
}

func (g *Gradients) Get(row, col uint32) Gradient {
	if row >= g.Rows || col >= g.Cols {
		fmt.Println("Bad Address", col, g.Cols, row, g.Rows)
		panic("Bad Address")
	}
	return g.Data[col*g.Rows+row]
}

func (g *Gradients) Set(row, col uint32, grad Gradient) {
	if row >= g.Rows || col >= g.Cols {
		fmt.Println("Bad Address", col, g.Cols, row, g.Rows)
		panic("Bad Address")
	}
	g.Data[col*g.Rows+row] = grad
}

func (g *Gradients) Size() uint32 {
	return g.Rows * g.Cols
}

func (g *Gradients) Apply(m *Matrix) {
	for i := uint32(0); i < m.Size(); i++ {
		g.Data[i].Apply(&m.Data[i])
	}
}

func (g *Gradients) Values() []float32 {
	vs := make([]float32, g.Size())
	for i := uint32(0); i < g.Size(); i++ {
		vs[i] = g.Data[i].Value
	}

	return vs
}
