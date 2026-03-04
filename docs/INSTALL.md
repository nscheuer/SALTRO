# Installation
```bash
sudo apt update
sudo apt upgrade
sudo apt install cmake ccache
sudo apt install -y build-essential python3-dev
```

# Virtual Environment for Python interface
```bash
sudo apt install python3-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Compile
```bash
cmake -S . -B build -DSALTRO_BUILD_PYTHON=ON
cmake --build build -j
```

# Compile in Debug Mode
```bash
sudo apt install gdb
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DSALTRO_BUILD_PYTHON=ON
cmake --build build -j
```

# Test
```bash
cd build
ctest --output-on-failure
cd ../tests
pytest -q
```