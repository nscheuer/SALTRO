# Installation
```bash
sudo apt install cmake
sudo apt install ninja-build
sudo apt install ccache
```

# Compile
```bash
cmake -S . -B build -DSALTRO_BUILD_PYTHON=ON
cmake --build build
```