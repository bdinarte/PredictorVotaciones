# setup.py es un archivo necesario para publicar el proyecto
# y después usarlo con pip.
# Referencia: https://packaging.python.org/tutorials/distributing-packages/#setup-args

from setuptools import setup, find_packages

setup(
    name="tec.ic.ia.p1.g03",
    packages=find_packages(),
    description="Proyecto I de IA: Predictor de votaciones CR",
    version="1.0.0",
    author="Julian Salinas, Brandon Dinarte, Armando López",
    license="GNU General Public License v3.0",
    keywords=['tec', 'ic', 'ia', 'p1', 'g03'],
    install_requires=['pandas', 'matplotlib', 'pytest', 'scipy', 'tensorflow', 'sklearn'],
    python_requires='>=3',
    include_package_data=True,
    package_data={'tec': ['*.txt', '*.csv', '.xlsx']}
)
