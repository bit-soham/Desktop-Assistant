import json
import subprocess
import difflib
import shlex
import os

def load_paths(json_file="user_data/exec_paths.json"):
    with open(json_file, "r") as f:
        return json.load(f)

def app_launch(command: str, json_file="user_data/exec_paths.json"):
    exe_dict = load_paths(json_file)

    parts = shlex.split(command)
    keyword = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []

    # Direct match
    if keyword in exe_dict:
        return _try_paths(exe_dict[keyword], keyword, args)

    # Fuzzy match
    similar = difflib.get_close_matches(keyword, exe_dict.keys(), n=1, cutoff=0.3)
    if similar:
        best = similar[0]
        return _try_paths(exe_dict[best], best, args)

    print(f"No match for '{keyword}'.")
    return None


def _try_paths(paths, name, args):
    # Normalize: string → list
    if isinstance(paths, str):
        paths = [paths]

    # Remove duplicates
    unique_paths = list(dict.fromkeys(paths))

    for path in unique_paths:
        if _run_if_exists(path, args):
            return path

    print(f"'{name}' not installed.")
    return None


def _run_if_exists(path, args):
    if not os.path.exists(path):
        return False

    try:
        subprocess.Popen([path] + args)
        return True
    except:
        return False
