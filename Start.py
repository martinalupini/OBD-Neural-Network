import subprocess
import sys
import importlib


# Funzione per verificare se pip è installato
def check_pip():
    try:
        import pip
        print("pip è già installato.")
    except ImportError:
        print("pip non è installato. Installazione in corso...")
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        print("pip installato correttamente.")


# Funzione per installare un pacchetto Python
def install_package(package_mod, package_nm):
    try:
        importlib.import_module(package_nm)
        print(f"{package_nm} è già installato.")
    except ImportError:
        print(f"{package_nm} non è installato. Installazione in corso...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_mod])


# Verifica se pip è installato
check_pip()

# Elenco dei pacchetti da controllare
packages = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scikit-learn": "scikit-learn",
    "pandas": "pandas"
}

# Verifica e installa i pacchetti necessari
for package_name, module_name in packages.items():
    install_package(module_name, package_name)

print("Tutti i pacchetti sono stati verificati e installati se necessario.")
