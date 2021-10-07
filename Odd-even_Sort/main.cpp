#include <stdio.h>
#include <string.h> 
#include <mpi.h> 
#include <iostream>
#include <time.h>
#include <algorithm>


int Compute_partner(int phase, int my_rank, int comm_sz) {
	int partner = -1;
	// Fase par
	if (phase % 2 == 0) {
		// Numero de proceso impar
		if ((my_rank % 2) != 0) {
			partner = my_rank - 1;
		} 
		else {
			partner = my_rank + 1;
		}
	} // Fase impar
	else {
		// Numero de proceso impar
		if ((my_rank % 2) != 0) {
			partner = my_rank + 1;
		} // Numero de proceso par
		else {
			partner = my_rank - 1;
		}
	}
	// Numero de proceso partner invalido
	if (partner == -1 || partner == comm_sz) {
		partner = -2;
	}
	return partner;
}

void select_less(int local_vector[], int local_partner[], int local_tmp[], int local_n) {
	int i = 0, j = 0;
	int k = 0;
	while (k < local_n) {
		if (local_vector[i] > local_partner[j]) {
			local_tmp[k] = local_partner[j];
			j++;
		} else {
			local_tmp[k] = local_vector[i];
			i++;
		}
		k++;
	}
}

void select_greater(int local_vector[], int local_partner[], int local_tmp[], int local_n) {
	int i = local_n - 1, j = local_n - 1;
	int k = local_n - 1;
	while (k >= 0)	{
		if (local_vector[i] < local_partner[j]) {
			local_tmp[k] = local_partner[j];
			j--;
		} else {
			local_tmp[k] = local_vector[i];
			i--;
		}
		k--;
	}
}

void odd_even_sort(int local_vec[], int n, int local_n, int my_rank, int comm_sz, MPI_Comm comm) {
	// Para almacenar los elementos del partner de my_rank
	int* local_partner;
	local_partner = (int*)malloc(local_n * sizeof(int));

	int* local_tmp;
	local_tmp = (int*)malloc(local_n * sizeof(int));

	for (int phase = 0; phase < comm_sz; phase++) {
		int partner = Compute_partner(phase, my_rank, comm_sz);
		if (partner != -2) {
			// Mantener los elementos menores
			if (my_rank < partner) {
				// Enviar los elementos de my_rank a partner
				MPI_Send(local_vec, local_n, MPI_INT, partner, 0, comm);
				// Recibir los elementos del partner de my_rank
				MPI_Recv(local_partner, local_n, MPI_INT, partner, 0, comm, MPI_STATUS_IGNORE);
				select_less(local_vec, local_partner, local_tmp, local_n);
			} // Mantener los elementos mayores
			else {
				// Recibir los elementos del partner de my_rank
				MPI_Recv(local_partner, local_n, MPI_INT, partner, 0, comm, MPI_STATUS_IGNORE);
				// Enviar los elementos de my_rank a partner
				MPI_Send(local_vec, local_n, MPI_INT, partner, 0, comm);
				select_greater(local_vec, local_partner, local_tmp, local_n);
			}
			// Actualizando los elementos locales
			for (int j = 0; j < local_n; j++) {
				local_vec[j] = local_tmp[j];				
			}
		}
	}
	free(local_partner);
	free(local_tmp);
}


int main(void) {

	int comm_sz;
	int my_rank;

	MPI_Init(NULL, NULL);

	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	int * local_vec;
	int n = 0, local_n = 0;

	// Ingresando el tamaño del vector
	if (my_rank == 0) {
		std::cout << "Ingrese el numero de elementos del vector: " << std::endl;
		std::cin >> n;
	}

	// Enviando a todos los procesos el tamaño del vector
	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Numero de elementos que cada proceso va a trabajar
	local_n = n / comm_sz;

	// Vector para cada proceso
	local_vec = (int*)malloc(local_n * sizeof(int));

	int* vec = NULL;

	// Ingresando los elementos del vector
	if (my_rank == 0) {
		vec = (int*)malloc(n * sizeof(int));
		srand(time(0));
		for (int i = 0; i < n; i++) {
			vec[i] = rand() % 100;
			//std::cin >> vec[i];
		}

		// Imprimiendo el vector generado
		/*
		std::cout << "Vector original: " << std::endl;
		for (int i = 0; i < n; i++) {
			std::cout << vec[i] << " ";
		}
		std::cout << std::endl;
		*/

		// Repartiendo a cada proceso los elementos con los que va a trabajar
		MPI_Scatter(vec, local_n, MPI_INT, local_vec, local_n, MPI_INT, 0, MPI_COMM_WORLD);
		free(vec);
	}
	else {
		MPI_Scatter(vec, local_n, MPI_INT, local_vec, local_n, MPI_INT, 0, MPI_COMM_WORLD);
	}

	// Primer ordenamiento para los elementos locales de cada proceso
	std::sort(local_vec, local_vec + local_n);

	double local_start, local_finish, local_elapsed, elapsed;
	MPI_Barrier(MPI_COMM_WORLD);
	local_start = MPI_Wtime();

	// Ordenamiento
	odd_even_sort(local_vec, n, local_n, my_rank, comm_sz, MPI_COMM_WORLD);

	local_finish = MPI_Wtime();
	local_elapsed = local_finish - local_start;
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


	int * final_vec = NULL;
	if (my_rank == 0) {
		final_vec = (int*)malloc(n * sizeof(int));
		MPI_Gather(local_vec, local_n, MPI_INT, final_vec, local_n, MPI_INT, 0, MPI_COMM_WORLD);
		/*
		std::cout << "Vector final: " << std::endl;
		for (int i = 0; i < n; i++) {
			std::cout << final_vec[i] << " ";
		}
		std::cout << std::endl;
		*/
		free(final_vec);
		std::cout << "Tiempo transcurrido: " << elapsed << std::endl;
	}
	else {
		MPI_Gather(local_vec, local_n, MPI_INT, final_vec, local_n, MPI_INT, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}