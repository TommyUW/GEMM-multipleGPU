main: main.o multiply.o
	mpic++ main.o multiply.o -lcudart -L/usr/local/cuda/lib64 -o program
multiply.o: multiply.cu
	nvcc -c multiply.cu -o multiply.o
main.o: main.cpp
	mpic++ -c main.cpp -o main.o
	
