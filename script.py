import fdtd
import fdtd.backend as bd
from fast_fdtd import FastFDTD 

import numpy as np
import matplotlib.pyplot as plt


def run_pybind11_simulation(N, grid_size):
    obj = FastFDTD()
    obj.setGridSize(grid_size[0], grid_size[1],
                    np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size),
                    np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size),
                    np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size))
    obj.setSimulationParameters(1.0, 0.1)
    obj.setSource(0.1, 0.2, 2.0, 1.0)

    result = obj.run(N)
    return result


def build_native_fdtd_simulation(N, grid_size):
    fdtd.set_backend("numpy")

    WAVELENGTH = 1550e-9
    SPEED_LIGHT: float = 299_792_458.0

    grid = fdtd.Grid(
        (2.5e-5, 1.5e-5, 1),
        grid_spacing=0.1 * WAVELENGTH,
        permittivity=1.0,
        permeability=1.0,
    )

    grid[100, 60, 0] = fdtd.PointSource(
        period=WAVELENGTH / SPEED_LIGHT, name="pointsource",
    )
    
    return grid

def compare_simulations(N, grid_size):
    result_pybind11 = run_pybind11_simulation(N, grid_size)
    native_fdtd = build_native_fdtd_simulation(N, grid_size)

    native_fdtd.visualize(
        native_fdtd,
        x=None,
        y=None,
        z=None,
        cmap="Blues",
        pbcolor="C3",
        pmlcolor=(0, 0, 0, 0.1),
        objcolor=(1, 0, 0, 0.1),
        srccolor="C0",
        detcolor="C2",
        show=True,
    )

    plt.figure(figsize=(10, 6))
    plt.plot(result_pybind11, label='Pybind11 2D FDTD', linestyle='--', marker='o')
    #plt.plot(result_native_fdtd, label='Native Python FDTD', linestyle='-', marker='x')
    plt.xlabel('Grid Index')
    plt.ylabel('Electric Field Value')
    plt.title('Comparison of Pybind11 and Native Python 2D FDTD')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    N = 100  
    grid_size = (300, 300)  

    compare_simulations(N, grid_size)

