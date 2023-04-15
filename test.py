import os

# název složky
dir_name = "slozka"

# vytvoření složky
os.makedirs(dir_name, exist_ok=True)

# získání cesty k složce
path = os.path.abspath(dir_name)

# vypsání cesty
print(path)
