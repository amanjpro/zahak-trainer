package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
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

func countSamples(paths []string) int64 {
	fmt.Printf("Paths to load %s\n", paths)
	totalCount := int64(0)
	countLines := func(f *os.File) int64 {
		count := int64(0)
		input := bufio.NewScanner(f)
		for input.Scan() {
			count++
			if err := input.Err(); err != nil {
				panic(err)
			}
		}

		return count
	}
	for _, path := range paths {
		file, err := os.Open(path)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		totalCount += countLines(file)
	}

	fmt.Printf("Loading %d samples\n", totalCount)

	return totalCount
}

func LoadDataset(paths string) []Data {
	pathsArray := strings.Split(paths, ",")
	data := make([]Data, countSamples(pathsArray))
	samples := 0
	for _, path := range pathsArray {
		file, err := os.Open(path)
		if err != nil {
			panic(err)
		}
		defer file.Close()

		reader := bufio.NewReader(file)
		skipNext := false
		for true {
			buf, pre, err := reader.ReadLine()
			if errors.Is(err, io.EOF) {
				break
			} else if err != nil {
				panic(err)
			}
			if pre {
				continue
			} else if skipNext {
				skipNext = false
				continue
			}
			skipNext = pre
			line := string(buf)
			if line == "" {
				break
			}
			sample := ParseLine(line)
			data[samples] = sample
			samples += 1
		}
	}

	runtime.GC()

	return data
}

func ParseLine(line string) Data {
	startIndex := 0
	endIndex := strings.Index(line, ";")
	if endIndex == -1 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	pos, wtm := FromFen(line[:endIndex])

	startIndex = endIndex + 7
	endIndex = strings.Index(line, ";eval")
	if endIndex == -1 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	score, err := strconv.Atoi(line[startIndex:endIndex])
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n%s\n", line, err))
	}

	normalizedScore := Sigmoid(float32(score))

	startIndex = strings.Index(line, ";outcome:") + 9
	if startIndex == -1 {
		panic(fmt.Sprintf("Bad line %s\n%s\n", line, err))
	}
	outcome, err := strconv.ParseFloat(line[startIndex:], 32)
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n%s\n", line, err))
	}

	if !wtm {
		score *= -1
		if outcome == 0 {
			outcome = 1
		} else if outcome == 1 {
			outcome = 0
		}
	}

	return Data{
		Input:   pos,
		Score:   normalizedScore,
		Outcome: float32(outcome),
	}
}
