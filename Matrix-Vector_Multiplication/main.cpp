#include <stdio.h>
#include <string.h> 
#include <mpi.h> 
#include <iostream>
#include <time.h>

void Mat_vect_mult(double local_A[], double local_x[], double local_y[], int local_m, int n, int local_n, MPI_Comm comm) {
	double* x;
	int local_i, j;
	int local_ok = 1;

	x = (double*)malloc(n * sizeof(double));

	MPI_Allgather(local_x, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, comm);

	for (local_i = 0; local_i < local_m; local_i++) {
		local_y[local_i] = 0.0;
		for (j = 0; j < n; j++) {
			local_y[local_i] += local_A[local_i * n + j] * x[j];
		}
	}
	free(x);
}



int main(void) {

	int comm_sz;
	int my_rank;

	MPI_Init(NULL, NULL);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	double* local_A;
	double* local_x;
	double* local_y;

	int m = 0, n = 0;
	int local_m = 0, local_n = 0;

	if (my_rank == 0) {
		std::cout << "Ingrese el numero de filas: " << std::endl;
		std::cin >> m;
		std::cout << "Ingrese el numero de columnas: " << std::endl;
		std::cin >> n;
	}
	
	// Enviando las dimensiones de la matriz a todos los procesos
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	local_m = m / comm_sz;
	local_n = n / comm_sz;

	local_A = (double*)malloc(local_m * n * sizeof(double));
	local_x = (double*)malloc(local_n * sizeof(double));
	local_y = (double*)malloc(local_m * sizeof(double));

	double* A = NULL;

	// Obteniendo los elementos de la matriz
	if (my_rank == 0) {
		A = (double*)malloc(m * n * sizeof(double));

		srand(time(0));
		
		for (int i = 0; i < m * n; i++) {
			A[i] = (double)(rand() % 10);
			/*
			if (i % n == 0 && i > 0) {
				std::cout << std::endl;
			}
			std::cout << A[i] << " ";
			*/
		}
		// std::cout << std::endl;
				
		// Enviando los componentes de la matriz a cada proceso
		MPI_Scatter(A, local_m * n, MPI_DOUBLE, local_A, local_m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		free(A);

	}
	else {
		MPI_Scatter(A, local_m * n, MPI_DOUBLE, local_A, local_m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}


	// Obteniendo los elementos del vector
	double * vec = NULL;
	if (my_rank == 0) {
		vec = (double*)malloc(n * sizeof(double));
		srand(time(0));
		for (int i = 0; i < n; i++) {
			vec[i] = (double)(rand() % 10);
			// std::cout << vec[i] << " ";
		}
		// std::cout << std::endl;
		MPI_Scatter(vec, local_n, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		free(vec);
	}
	else {
		MPI_Scatter(vec, local_n, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

	// Tomando tiempos
	double local_start, local_finish, local_elapsed, elapsed;

	MPI_Barrier(MPI_COMM_WORLD);

	local_start = MPI_Wtime();

	// Funcion de multiplicacion matriz vector

	Mat_vect_mult(local_A, local_x, local_y, local_m, n, local_n, MPI_COMM_WORLD);

	local_finish = MPI_Wtime();
	local_elapsed = local_finish - local_start;

	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	double* final_vec = NULL;
	if (my_rank == 0) {
		final_vec = (double*)malloc(n * sizeof(double));
		MPI_Gather(local_y, local_m, MPI_DOUBLE, final_vec, local_m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		/*
		for (int i = 0; i < n; i++) {
			std::cout << final_vec[i] << " ";
		}
		std::cout << std::endl;
		*/
		free(final_vec);
		std::cout << "Tiempo transcurrido: " << elapsed << std::endl;
	}
	else {
		MPI_Gather(local_y, local_m, MPI_DOUBLE, final_vec, local_m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

	free(local_A);
	free(local_x);
	free(local_y);

	MPI_Finalize();
	return 0;
}



