struct Param {

  double Hc;
  double Jc;
  double Ec;

  double t;

  int source_i;
  int source_j;
  double tau; // duration of excitation
  double omega; // frequency of excitation
  double dt; // frequency of excitation
};


void copyToDevice(element_matriz &arr) {
  cudaMemcpy(arr.d_data, arr.h_data, arr.length*sizeof(double), cudaMemcpyHostToDevice);
}

void copyToHost(element_matriz &arr) {
  cudaMemcpy(arr.h_data, arr.d_data, arr.length*sizeof(double), cudaMemcpyDeviceToHost);
}

void destruct(element_matriz &arr){
  cudaFree(arr.d_data);
}

void destruct(Tensor& tensor)
{
  destruct(tensor.Hx);
  destruct(tensor.Ey);
  destruct(tensor.Hz);
}

__device__
double Get(const element_matriz arr, int i_0, int i_1) {
    
    return arr.d_data[i_0 * arr.size_1 + i_1];
}

__device__
void Set(element_matriz arr, int i_0, int i_1, double value) {

    arr.d_data[i_0 * arr.size_1 + i_1] = value;
}

__device__
double Get(const element_matriz arr, int i_0, int i_1) {

    return arr.d_data[i_0 * arr.size_1 + i_1];
}
