# FastFDTD

FDTD utilizando CUDA e PyBind11

1. Criar diretório de build e fazer a build do projeto

mkdir build
cd build
cmake ..

2. Realizar o setup 

Ativar o ambiente virtual python local ('python -m venv <directorio>')
pip install . (na raiz)

3. Importar o modulo fast_fdtd em programas Python no mesmo ambiente.

import fast_fdtd
from fast_fdtd import FastFDTD