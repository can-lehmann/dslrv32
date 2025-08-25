RV32I_CFLAGS := -fPIC -ffreestanding -nostdlib -target riscv32-unknown-unknown -march=rv32i

sim: waveform.fst

rv32i_%.o: rv32i_%.c
	clang -c -O1 ${RV32I_CFLAGS} -o $@ $^

rv32i_%: rv32i_%.o
	ld.lld -o $@ $^

Processor.v: main.py rv32i_test
	python3 main.py

Testbench.vvp: Testbench.v Processor.v
	iverilog -o $@ $^

waveform.fst: Testbench.vvp
	vvp $< -fst

%.svg: %.gv
	dot -Tsvg -o $@ $<
