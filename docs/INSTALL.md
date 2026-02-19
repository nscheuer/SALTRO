# Installation
```bash
sudo apt install cmake
sudo apt install ninja-build
sudo apt install ccache
```

# Compile
```bash
cmake -S . -B build -DSALTRO_BUILD_PYTHON=ON
cmake --build build -j
```

# Python Tests
```bash
pip install git+https://github.com/poliastro/poliastro
pip install astropy
```