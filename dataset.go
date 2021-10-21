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
		Score   int16
		Outcome int8
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

	pos := FromFen(line[:endIndex])
	wm := pos[len(pos)-1] == 768

	startIndex = endIndex + 7
	endIndex = strings.Index(line, ";eval")
	if endIndex == -1 {
		panic(fmt.Sprintf("Bad line %s\n", line))
	}

	score, err := strconv.Atoi(line[startIndex:endIndex])
	if err != nil {
		panic(fmt.Sprintf("Bad line %s\n%s\n", line, err))
	}

	startIndex = strings.Index(line, ";outcome:") + 9
	if startIndex == -1 {
		panic(fmt.Sprintf("Bad line %s\n%s\n", line, err))
	}
	var outcome int8
	if line[startIndex:] == "0.0" {
		outcome = 0
	} else if line[startIndex:] == "1.0" {
		outcome = 2
	} else {
		outcome = 1
	}

	if !wm {
		score *= -1
		if outcome == 2 {
			outcome = 0
		} else if outcome == 0 {
			outcome = 2
		}
	}

	return Data{
		Input:   pos,
		Score:   int16(score),
		Outcome: outcome,
	}
}
