package main

import (
	"math"
)

const (
	CostEvalWeight float32 = 0.5
	CostWDLWeight  float32 = 1.0 - CostEvalWeight
)

func Sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(SigmoidScale*(-x)))))
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

func CalculateCostGradient(output, evalTarget, wdlTarget float32) float32 {
	return 2.0*CostEvalWeight*(output-evalTarget) + 2.0*CostWDLWeight*(output-wdlTarget)
}
