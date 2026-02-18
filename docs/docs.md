# Virtual Environment Instructions
```bash
python3 -m venv venv
pip install breathe pydata_sphinx_theme
```

# Doxygen Instructions
```bash
sudo apt install doxygen
source venv/bin/activate
cd docs/
doxygen
```

# Sphinx Instructions
```bash
python3 generate_rst.py
python3 -m sphinx -W -b html source build
```