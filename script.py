import fdtd
import fast_fdtd  

import numpy as np
import matplotlib.pyplot as plt


def run_pybind11_simulation(N, grid_size):
    obj = fast_fdtd.FastFDTD()
    obj.setGridSize(grid_size[0], grid_size[1],
                    np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size),
                    np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size),
                    np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size))
    obj.setSimulationParameters(1.0, 0.1)
    obj.setSource(0.1, 0.2, 2.0, 1.0)

    result = obj.run(N)
    return result


def run_native_fdtd_simulation(N, grid_size):
    
    sim = fdtd.Simulation(
        size=grid_size,
        resolution=1e-6,
        CFL=0.99,
        geometry=[fdtd.PointMaterial(), fdtd.PointMaterial()],
        Courant=0.5
    )

    sim.add_source(fdtd.Gaussian1d(frequency=1e14, fwidth=1e13))

    for _ in range(N):
        sim.run(total_time=1e-14)

    result = np.array(sim.electric_field[0].data)

    return result

def compare_simulations(N, grid_size):
    result_pybind11 = run_pybind11_simulation(N, grid_size)
    result_native_fdtd = run_native_fdtd_simulation(N, grid_size)


    plt.figure(figsize=(10, 6))
    plt.plot(result_pybind11, label='Pybind11 2D FDTD', linestyle='--', marker='o')
    plt.plot(result_native_fdtd, label='Native Python FDTD', linestyle='-', marker='x')
    plt.xlabel('Grid Index')
    plt.ylabel('Electric Field Value')
    plt.title('Comparison of Pybind11 and Native Python 2D FDTD')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    N = 100  # Number of timesteps
    grid_size = (300, 300)  # Size of the 2D simulation grid

    compare_simulations(N, grid_size)

