package main

import (
	"testing"
)

func TestParseLine(t *testing.T) {
	line := "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1;score:-0.720000;eval:50;qs:0;outcome:0.5"

	data := ParseLine(line)

	if data.Score != Sigmoid(-72) {
		t.Errorf("Score is parsed wrong, expected %f, got %f", Sigmoid(-72), data.Score)
	}
	if data.Outcome != 0.5 {
		t.Errorf("Outcome is parsed wrong, expected %f, got %f", 0.5, data.Outcome)
	}
	expected := []int16{
		632, 505, 570, 699, 764, 573, 510, 639,
		432, 433, 434, 435, 436, 437, 438, 439,
		8, 9, 10, 11, 12, 13, 14, 15,
		192, 65, 130, 259, 324, 133, 70, 199,
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
