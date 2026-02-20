# Virtual Environment Instructions
```bash
python3 -m venv venv
pip install breathe pydata_sphinx_theme
sudo apt install doxygen
source venv/bin/activate
```

# Compile Instructions
```bash
cd docs/
doxygen
python3 generate_rst.py
python3 -m sphinx -W -b html source build
```

# Delete instructions
Remove docs/build
Remove docs/source/api
Remove docs/source/html
Remove docs/source/latex
Remove docs/source/xml