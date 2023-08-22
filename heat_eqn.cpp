#include "heat_eqn.hpp"

/**
 * Constructor for the HeatSimulation class.
 * Initializes MPI, sets up the MPI grid topology, and prepares data types for communication.
 *
 * @param argc      The command-line argument count.
 * @param argv      The command-line arguments.
 */
HeatSimulation::HeatSimulation(int argc, char* argv[])
		: grid((nx+2) * (ny+2), 0), dims{0,0}, A((nx+2) * (ny+2), 0)
 {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    nworkers = nprocs - 1;
	MPI_Dims_create(nworkers, 2, dims.data());


    nxLocal = nx / dims[0];
    nyLocal = ny / dims[1];

    MPI_Type_vector(nxLocal+2, 1, nyLocal+2, MPI_DOUBLE, &colA);
    MPI_Type_commit(&colA);
    MPI_Type_vector(nxLocal, 1, nyLocal, MPI_DOUBLE, &colB);
    MPI_Type_commit(&colB);
    MPI_Type_vector(nyLocal+2, 1, 1, MPI_DOUBLE, &rowA);
    MPI_Type_commit(&rowA);
    MPI_Type_vector(nyLocal, 1, 1, MPI_DOUBLE, &rowB);
    MPI_Type_commit(&rowB);
    MPI_Type_vector(nxLocal+2, nyLocal+2, ny+2, MPI_DOUBLE, &matrixA);
    MPI_Type_commit(&matrixA);
    MPI_Type_vector(nxLocal, nyLocal, ny, MPI_DOUBLE, &matrixB);
    MPI_Type_commit(&matrixB);
}


/**
 * Initialize the grid with initial temperature values based on the initial_heat function.
 * This function populates the input grid data with initial temperature values
 * calculated using the benchmark function for each grid cell.
 *
 * @param x         The total number of rows in the grid.
 * @param y         The total number of columns in the grid.
 * @param data      The grid data to be initialized.
 */
void HeatSimulation::initialise_grid(int x, int y) {
	int i, j;
	for(i=0; i<x; i++)
	{
		for(j=0; j<y; j++)
			grid[i*y+j] = initial_heat(i, j, 0);
	}
}


/**
 * Update the temperature value of a grid cell using the Jacobi method.
 * This function calculates the new temperature value at the specified grid
 * cell based on neighboring cells and the heat equation.
 *
 * @param x         The total number of rows in the grid.
 * @param y         The total number of columns in the grid.
 * @param i         The row index of the grid cell being updated.
 * @param j         The column index of the grid cell being updated.
 * @param t         The time step.
 * @param A         The input matrix.
 * @param B         The output matrix.
 * @param id        The ID of the current MPI process.
 */

void HeatSimulation::update(int x, int y, int i, int j, int t, std::vector<double>& A, std::vector<double>& B, int id){
	double sumOfNeighbours = A[(i-1)*y+j] + A[(i+1)*y+j] + A[i*y+(j-1)] + A[i*y+(j+1)];
	double heatEqVal = (x_diff*x_diff)*f(i+floor((id-1)/dims[1])*(x-2), j+((id-1)%dims[1])*(y-2), 0);
	B[(i-1)*(y-2)+(j-1)] = (sumOfNeighbours + heatEqVal) /4.0;
}


/**
 * Update the boundaries of the matrix A with Dirichlet boundary conditions.
 * This function sets the boundary conditions of the
 * grid matrix A.
 *
 * @param x         The total number of rows in the grid.
 * @param y         The total number of columns in the grid.
 * @param t         The time step.
 * @param A         The matrix to be updated with boundary conditions.
 * @param B         The matrix containing updated temperature values.
 * @param id        The ID of the current MPI process.
 */
void HeatSimulation::assign_boundaries(int x, int y, int t, std::vector<double>& A, std::vector<double>& B, int id) {
	int i, j;
	for(i=1; i<x-1; i++)
	{
		for(j=1; j<y-1; j++)
			A[i*y+j] = B[(i-1)*(y-2)+(j-1)];
	}

	/* Set the boundary values of A based on the estimated  */
	if(floor((id-1)/dims[1]) == 0)
	{
		for(j=0; j<y; j++)
			A[0*y+j] = initial_heat(0, j+((id-1)%dims[1])*(y-2), t);
	}
	if(floor((id-1)/dims[1]) == dims[0]-1)
	{
		for(j=0; j<y; j++)
			A[(x-1)*y+j] = initial_heat(nx+1, j+((id-1)%dims[1])*(y-2), t);
	}
	if(id%dims[1] == 1 || dims[1] == 1)
	{
		for(i=0; i<x; i++)
			A[i*y+0] = initial_heat(i+floor((id-1)/dims[1])*(x-2), 0, t);
	}
	if(id%dims[1] == 0)
	{
		for(i=0; i<x; i++)
			A[i*y+(y-1)] = initial_heat(i+floor((id-1)/dims[1])*(x-2), ny+1, t);
	}
}

void HeatSimulation::print_to_file(int x, int y, int n){
    std::ofstream file;
    char name[13];

    sprintf(name, "heat_mpiN%i.txt", n);
    file.open(name);

    for (int i = 0; i < nxLocal; i++) {
        for (int j = 0; j < nyLocal; j++) {
            file << grid[i * nyLocal + j] << " ";
        }
        file << "; \n";
    }

    file.close();
}


/**
 * Calculate the right-hand side of the heat equation at a specific grid cell.
 * This function incorporates the initial temperature
 * distribution function to model the temperature change over time.
 *
 * @param x         The x-coordinate of the grid.
 * @param y         The y-coordinate of the grid.
 * @param t         The time step.
 * @return          The value representing the change in temperature over time
 *                  at the specified cell.
 */

double HeatSimulation::f(int x, int y, int t) {
    return (x_diff*x_diff + y_diff * y_diff-t_diff) * initial_heat(x, y, t);
}


/**
 * Calculate the initial temperature distribution at a specific grid cell.
 * This function is based on the heat equation and represents the temperature
 * distribution at time t = 0.
 *
 * @param x     The x-coordinate of the grid cell.
 * @param y     The y-coordinate of the grid cell.
 * @param t     The time step.
 * @return      The initial temperature value at the specified cell.
 */
double HeatSimulation::initial_heat(int x, int y, int t) {
	return sin(x*x_diff)*cos(y * y_diff)*exp(-t*t_diff);
}


/**
 * Run the heat simulation using the parallel Jacobi method.
 * This function coordinates the entire simulation.
 */
void HeatSimulation::run() {
int i, j;
if(id == MASTER)
{
	std::cout << nworkers << " Workers, Grid Structure " << dims[0] << "x" << dims[1] << "\n";

	tstart = MPI_Wtime();
	initialise_grid(nx+2, ny+2);

	/* Send full grid to neighbours incl. ghost cells */
	message = BEGIN;
	for(i=1; i<=dims[0]; i++)
	{
		for(j=1; j<=dims[1]; j++)
		{
			dest = (i-1)*dims[1]+j;
			MPI_Send(&grid[(i-1)*nxLocal*(ny+2)+(j-1)*nyLocal], 1, matrixA, dest, message, MPI_COMM_WORLD);
		}
	}


	/* Receive full grid incl. ghost cells */
	message = END;
	for(i=1; i<=dims[0]; i++)
	{
		for(j=1; j<=dims[1]; j++)
		{
		source = (i-1)*dims[1]+j;
		MPI_Recv(&grid[(i-1)*nxLocal*ny+(j-1)*(nyLocal)], 1, matrixB, source, message, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
	}

	tend = MPI_Wtime();
    std::cout << "Time taken = " << tend - tstart << " seconds" << std::endl;

	print_to_file(nx, ny, 1);
	MPI_Finalize();
}

if(id != MASTER)
{
	/* Only initialize the amount of memory needed on the slave process */
	int t;
	std::vector<double> A((nxLocal+2) * (nyLocal+2));
	std::vector<double> B(nxLocal*nyLocal);

	source = MASTER;
	message = BEGIN;
	MPI_Recv(&A[0], (nxLocal+2)*(nyLocal+2), MPI_DOUBLE, source, message, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	message = COMM;
	left = id-1;
	right = id+1;
	up = id-dims[1];
	down = id+dims[1];

	/* Handle the ghost column edge cases */
	if(left < 1 || left > nworkers || id%dims[1] == 1 || dims[1] == 1)
		left = MPI_PROC_NULL;
	if(right < 1 || right > nworkers || id%dims[1] == 0)
		right = MPI_PROC_NULL;
	if(up < 1 || up > nworkers)
		up = MPI_PROC_NULL;
	if(down < 1 || down > nworkers)
		down = MPI_PROC_NULL;

	/* Begin updating the sides of B */
	for(t=1; t<=time_steps; t++)
	{
		if(id == 1 && (t%(time_steps/10)) == 0)
			std::cout << "Time Step= " << t << "\n";

	for(j=1; j<nyLocal+1; j++)
	{
		i = 1;
		update(nxLocal+2, nyLocal+2, i, j, t, A, B, id);
		i = nxLocal;
		update(nxLocal+2, nyLocal+2, i, j, t, A, B, id);
	}
	for(i=1; i<nxLocal+1; i++)
	{
		j = 1;
		update(nxLocal+2, nyLocal+2, i, j, t, A, B, id);
		j = nyLocal;
		update(nxLocal+2, nyLocal+2, i, j, t, A, B, id);
	}

	/* Update A based on the new updated B using asynchronous communication*/
	MPI_Isend(&B[0], 1, rowB, up, message, MPI_COMM_WORLD, &request[0]);  								// B(Row 1) -> A(Ghost)
	MPI_Isend(&B[(nxLocal-1)*nyLocal], 1, rowB, down, message, MPI_COMM_WORLD, &request[1]); 			// B(Row n-1) -> A(Ghost)
	MPI_Isend(&B[0], 1, colB, left, message, MPI_COMM_WORLD, &request[2]); 								// B(left_col) -> A(Ghost)
	MPI_Isend(&B[nyLocal-1], 1, colB, right, message, MPI_COMM_WORLD, &request[3]); 					// B(right_col) -> A(Ghost)

	/* Receive into A */
	MPI_Irecv(&A[1], 1, rowA, up, message, MPI_COMM_WORLD, &request[4]);		
	MPI_Irecv(&A[(nxLocal+2-1)*(nyLocal+2)+1], 1, rowA, down, message, MPI_COMM_WORLD, &request[5]);
	MPI_Irecv(&A[1*(nyLocal+2)], 1, colA, left, message, MPI_COMM_WORLD, &request[6]);
	MPI_Irecv(&A[1*(nyLocal+2)+nyLocal+2-1], 1, colA, right, message, MPI_COMM_WORLD, &request[7]);

	for(i=1+1; i<nxLocal+2-1-1; i++)
	{
		for(j=1+1; j<nyLocal+2-1-1; j++)
			update(nxLocal+2, nyLocal+2, i, j, t, A, B, id);
	}

	assign_boundaries(nxLocal+2, nyLocal+2, t, A, B, id);

	MPI_Waitall(8, request, MPI_STATUSES_IGNORE);
	}

	dest = MASTER;
	message = END;

	/* Accumulate the result on master */
	MPI_Send(&B[0], nxLocal*nyLocal, MPI_DOUBLE, dest, message, MPI_COMM_WORLD);

	MPI_Finalize();
}
}
