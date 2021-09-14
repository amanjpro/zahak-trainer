package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

const (
	CostEvalWeight float64 = 0.5
	CostWDLWeight  float64 = 1.0 - CostEvalWeight
)

func Sigmoid(a, b int, x float64) float64 {
	return 1.0 / (1.0 + math.Exp(SigmoidScale*(-x)))
}

func SigmoidPrime(a, b int, x float64) float64 {
	return x * (1.0 - x) * SigmoidScale

}

func ReLu(a, b int, x float64) float64 {
	return math.Max(x, 0.0)
}

func ReLuPrime(a, b int, x float64) float64 {
	if x > 0.0 {
		return 1.0
	}
	return 0.0
}

func CalculateCost(output *mat.Dense, evalTarget, wdlTarget float64) *mat.Dense {
	fn := func(a, b int, x float64) float64 {
		return (2.0*CostEvalWeight*(x-evalTarget) + 2.0*CostWDLWeight*(x-wdlTarget)*SigmoidPrime(0, 0, x))
	}
	return Apply(fn, output)
}
