PERF_FLAGS := -F 1000 -g --call-graph dwarf

EXEC_FLAGS := 80 2 4000 6 6

run:
	cargo run --release 

flame:
	cargo flamegraph -c "record $(PERF_FLAGS)" -- $(EXEC_FLAGS)

time:
	cargo build --release 
	hyperfine -i "target/release/lab4 $(EXEC_FLAGS)"
