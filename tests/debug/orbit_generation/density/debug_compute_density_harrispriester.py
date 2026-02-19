import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))
import saltro_py

from astropy import units as u
from poliastro.earth.atmosphere import COESA62, COESA76

Re = 6378.1363e3

alts = np.linspace(120e3, 700e3, 400)
N = len(alts)

R = np.zeros((3, N))
S = np.zeros((3, N))

for i, alt in enumerate(alts):
    r = Re + alt
    R[:, i] = np.array([r, 0.0, 0.0])

S[:] = np.array([[1.0], [0.0], [0.0]])

ok, rho_saltro = saltro_py.compute_density_harrispriester(R, S)
if not ok:
    raise RuntimeError("saltro HP failed")

coesa62 = COESA62()
coesa76 = COESA76()

rho_62 = []
rho_76 = []

for alt in alts:
    rho_62.append(coesa62.density(alt * u.m).to_value(u.kg / u.m**3))
    rho_76.append(coesa76.density(alt * u.m).to_value(u.kg / u.m**3))

rho_62 = np.array(rho_62)
rho_76 = np.array(rho_76)

plt.figure(figsize=(7,6))
plt.semilogy(alts/1e3, rho_saltro, linewidth=3, label="saltro harrispriester")
plt.semilogy(alts/1e3, rho_62, label="coesa62")
plt.semilogy(alts/1e3, rho_76, label="coesa76")
plt.xlabel("altitude km")
plt.ylabel("density kg m^-3")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()