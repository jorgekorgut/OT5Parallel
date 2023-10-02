main : naive reduce atomic divided vector
	@echo Done
naive: tp_openmp_part_1_pi.cpp
	g++ -o naive tp_openmp_part_1_pi.cpp -fopenmp -O3 -g -march=native
reduce: tp_openmp_part_1_pi_impl_reduce.cpp
	g++ -o reduce tp_openmp_part_1_pi_impl_reduce.cpp -fopenmp -O3 -g -march=native
atomic: tp_openmp_part_1_pi_impl_atomic.cpp
	g++ -o atomic tp_openmp_part_1_pi_impl_atomic.cpp -fopenmp -O3 -g -march=native
divided: tp_openmp_part_1_pi_impl_divided.cpp
	g++ -o divided tp_openmp_part_1_pi_impl_divided.cpp -fopenmp -O3 -g -march=native
vector: 
	g++ -o vector tp_openmp_part_2_vector.cpp -fopenmp -O3 -g -march=native