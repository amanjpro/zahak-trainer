package main

import (
	"math"
)

const (
	CostEvalWeight float32 = 0.5
	CostWDLWeight  float32 = 1.0 - CostEvalWeight
)

func Sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(SigmoidScale*-x))))
}

func SigmoidPrime(x float32) float32 {
	return x * (1.0 - x) * SigmoidScale
}

func ReLu(x float32) float32 {
	return float32(math.Max(float64(x), 0.0))
}

func ReLuPrime(x float32) float32 {
	if x > 0.0 {
		return 1.0
	}
	return 0.0
}

func CalculateCost(output, wdlTarget, evalTarget *Matrix) *Matrix {
	costMatrix := FromTemplate(output)
	for i := uint32(0); i < output.Size(); i++ {
		lhs := CostEvalWeight * float32(math.Pow(float64(output.Data[i]-evalTarget.Data[i]), 2.0))
		rhs := CostWDLWeight * float32(math.Pow(float64(output.Data[i]-wdlTarget.Data[i]), 2.0))
		costMatrix.Data[i] = lhs + rhs
	}
	return costMatrix
}
