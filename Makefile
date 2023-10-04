PART1_SRC=Part1
BIN=Executables

main : Part1
	@echo Done

Part1 : $(BIN)/naive.o $(BIN)/reduce.o $(BIN)/atomic.o $(BIN)/divided.o
	@echo Part1 Compiled

$(BIN)/naive.o: $(PART1_SRC)/tp_openmp_part_1_pi.cpp
	g++ -o $(BIN)/naive.o $(PART1_SRC)/tp_openmp_part_1_pi.cpp -fopenmp -O3 -g -march=native
$(BIN)/reduce.o: $(PART1_SRC)/tp_openmp_part_1_pi_impl_reduce.cpp
	g++ -o $(BIN)/reduce.o $(PART1_SRC)/tp_openmp_part_1_pi_impl_reduce.cpp -fopenmp -O3 -g -march=native
$(BIN)/atomic.o: $(PART1_SRC)/tp_openmp_part_1_pi_impl_atomic.cpp
	g++ -o $(BIN)/atomic.o $(PART1_SRC)/tp_openmp_part_1_pi_impl_atomic.cpp -fopenmp -O3 -g -march=native
$(BIN)/divided.o: $(PART1_SRC)/tp_openmp_part_1_pi_impl_divided.cpp
	g++ -o $(BIN)/divided.o $(PART1_SRC)/tp_openmp_part_1_pi_impl_divided.cpp -fopenmp -O3 -g -march=native



vector: 
	g++ -o vector tp_openmp_part_2_vector.cpp -fopenmp -O3 -g -march=native