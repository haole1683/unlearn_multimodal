from pathlib import Path

def create_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True) 
    
def join_path(path, *paths):
    return Path(path).joinpath(*paths)