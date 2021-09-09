package main

func main() {
	topology := NewTopology(14, 13, 12, 11)
	net := CreateNetwork(topology)
	net.SaveCheckpoint("/tmp/net")
	net = LoadCheckpoint("/tmp/net")
}
