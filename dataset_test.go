package main

import (
	"testing"
)

func TestParseLine(t *testing.T) {
	line := "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1;score:342;eval:351;qs:351;outcome:1.0"

	data := ParseLine(line)

	if data.Score != 342 {
		t.Errorf("Score is parsed wrong, expected %d, got %d", 342, data.Score)
	}
	if data.Outcome != 2 {
		t.Errorf("Outcome is parsed wrong, expected %d, got %d", 2, data.Outcome)
	}
	expected := []int16{
		632, 505, 570, 699, 764, 573, 510, 639,
		432, 433, 434, 435, 436, 437, 438, 439,
		8, 9, 10, 11, 12, 13, 14, 15,
		192, 65, 130, 259, 324, 133, 70, 199, 768,
	}
	if !sameArray16(data.Input, expected) {
		t.Errorf("Position is parsed wrong, expected %v, got %v", expected, data.Input)
	}
}

func sameArray16(expected, actual []int16) bool {
	if len(expected) != len(actual) {
		return false
	}
	for i := 0; i < len(expected); i++ {
		if expected[i] != actual[i] {
			return false
		}
	}
	return true
}
