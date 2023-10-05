PART1_SRC=Part1
PART2_SRC=Part2
BIN=Executables

#Compiling windows/linux
#GCC=x86_64-w64-mingw32-g++
GCC=g++

LIBS=-static-libstdc++ -static-libgcc -lgomp -static


main : Part1 Part2
	@echo Done

Part1 : $(BIN)/naive.o $(BIN)/reduce.o $(BIN)/atomic.o $(BIN)/divided.o
	@echo Part1 Compiled

$(BIN)/naive.o: $(PART1_SRC)/tp_openmp_part_1_pi.cpp
	$(GCC) -o $(BIN)/naive.o $(PART1_SRC)/tp_openmp_part_1_pi.cpp -fopenmp -O3 $(LIBS) -g -march=native
$(BIN)/reduce.o: $(PART1_SRC)/tp_openmp_part_1_pi_impl_reduce.cpp
	$(GCC) -o $(BIN)/reduce.o $(PART1_SRC)/tp_openmp_part_1_pi_impl_reduce.cpp -fopenmp $(LIBS) -O3 -g -march=native
$(BIN)/atomic.o: $(PART1_SRC)/tp_openmp_part_1_pi_impl_atomic.cpp
	$(GCC) -o $(BIN)/atomic.o $(PART1_SRC)/tp_openmp_part_1_pi_impl_atomic.cpp -fopenmp $(LIBS) -O3 -g -march=native
$(BIN)/divided.o: $(PART1_SRC)/tp_openmp_part_1_pi_impl_divided.cpp
	$(GCC) -o $(BIN)/divided.o $(PART1_SRC)/tp_openmp_part_1_pi_impl_divided.cpp -fopenmp $(LIBS) -O3 -g -march=native

Part2 : $(BIN)/vector_seq.o $(BIN)/vector_parallel.o $(BIN)/vector_simd.o
	@echo Part2 Compiled

$(BIN)/vector_seq.o: $(PART2_SRC)/tp_openmp_part_2_vector.cpp
	$(GCC) -o $(BIN)/vector_seq.o $(PART2_SRC)/tp_openmp_part_2_vector.cpp -fopenmp $(LIBS) -O3 -g -march=native

$(BIN)/vector_parallel.o: $(PART2_SRC)/tp_openmp_part_2_vector_parallel.cpp
	$(GCC) -o $(BIN)/vector_parallel.o $(PART2_SRC)/tp_openmp_part_2_vector_parallel.cpp -fopenmp $(LIBS) -O3  -march=native

$(BIN)/vector_simd.o: $(PART2_SRC)/tp_openmp_part_2_vector_simd.cpp
	$(GCC) -o $(BIN)/vector_simd.o $(PART2_SRC)/tp_openmp_part_2_vector_simd.cpp -fopenmp $(LIBS) -O3  -march=native