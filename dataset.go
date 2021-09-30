package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type (
	Data struct {
		Input   []int16
		Score   float32
		Outcome float32
	}
)

func LoadDataset(path string) []*Data {
	file, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data := make([]*Data, 0, 57_220_422)

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			break
		}
		sample := ParseLine(line)
		data = append(data, sample)
	}

	if err := scanner.Err(); err != nil {
		panic(err)
	}

	return data
}

func ParseLine(line string) *Data {
	startIndex := 0
	endIndex := strings.Index(line, ";")
	if endIndex == -1 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	pos := FromFen(strings.TrimSpace(line[:endIndex]))

	startIndex = endIndex + 7
	endIndex = strings.Index(line, ";eval")
	if endIndex == -1 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	score, err := strconv.Atoi(strings.TrimSpace(line[startIndex:endIndex]))
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n%s\n", line, err))
	}

	normalizedScore := Sigmoid(float32(score))

	startIndex = strings.Index(line, ";outcome:") + 9
	if startIndex == -1 {
		panic(fmt.Sprintf("Bad line %s\n%s\n", line, err))
	}
	outcome, err := strconv.ParseFloat(strings.TrimSpace(line[startIndex:]), 32)
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n%s\n", line, err))
	}

	return &Data{
		Input:   pos,
		Score:   normalizedScore,
		Outcome: float32(outcome),
	}
}

func clamp(lower, value, upper int) int {
	if value > upper {
		return upper
	}

	if value < lower {
		return lower
	}
	return value
}
