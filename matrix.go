package main

import "gonum.org/v1/gonum/mat"

func Dot(m, n mat.Matrix) *mat.Dense {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func Apply(fn func(i, j int, v float64) float64, m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

func Scale(s float64, m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func Multiply(m, n *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func Add(m, n *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}
