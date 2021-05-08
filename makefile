make:
	nvcc -std=c++17 main.cu -o main -O2 -lm -lX11