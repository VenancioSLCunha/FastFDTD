import "solver_func.h"

struct element_matriz {

    int tam_0;
    int tam_1;

    int length;
    
    double* h_data;
    double* d_data;

    void CopyToHost() {
        cudaMemcpy(h_data, d_data, length * sizeof(double), cudaMemcpyDeviceToHost);
    }

    void CopyToDevice() {
        cudaMemcpy(d_data, h_data, length * sizeof(double), cudaMemcpyHostToDevice);
    }
};

struct element_object {


}

struct Tensor {

  element_matriz Hx;
  element_matriz Ey;
  element_matriz Hz;

  void CopyToHost(){

    Hx.CopyToHost();
    Ey.CopyToHost();
    Hz.CopyToHost();
  }
  void CopyToDevice(){

    Hx.CopyToDevice();
    Ey.CopyToDevice();
    Hz.CopyToDevice();
  }
};

__device__
void converte_index(const element_matriz &array, const int raveled_index, int &i,int &j) {
  
    i= raveled_index / array.tam_1;
    j= raveled_index % array.tam_1;
}

void make_elemento(element_matriz &array, double* data, int tam_0, int tam_1) {
  array.tam_0=tam_0;
  array.tam_1=tam_1;
  array.length=tam_0*tam_1;
  
  cudaMalloc(&array.d_data,array.length*sizeof(double));
  
  array.h_data = data;
}

void make_elemento(Tensor& tensor, double*xdata,double*ydata,double*zdata,int tam_0,int tam_1) {

  make_elemento(tensor.Hx, xdata, tam_0, tam_1);
  make_elemento(tensor.Ey, ydata, tam_0, tam_1);
  make_elemento(tensor.Hz, zdata, tam_0, tam_1);
}
