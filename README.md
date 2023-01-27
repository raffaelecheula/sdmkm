# sdmkm

Tools for structure-dependent microkinetic modelling.

Installation:
```bash
git clone https://github.com/raffaelecheula/sdmkm.git sdmkm
cd sdmkm
conda create --name sdmkm --channel cantera cantera==2.5.1 ipython matplotlib jupyter ase pymatgen
conda activate sdmkm
pip install -e .
```