# device_picker_both.py
import re
import pyaudio # type: ignore
from collections import OrderedDict

def list_devices_info():
    p = pyaudio.PyAudio()
    try:
        devs = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            devs.append({
                "index": i,
                "name": info.get("name", "") or "",
                "maxInputChannels": int(info.get("maxInputChannels", 0)),
                "maxOutputChannels": int(info.get("maxOutputChannels", 0)),
                "defaultSampleRate": info.get("defaultSampleRate", 0)
            })
        return devs
    finally:
        p.terminate()

def friendly_base_name(raw_name):
    """Normalize and shorten verbose driver names into a friendly device name."""
    s = (raw_name or "").strip()
    # drop driver path suffix after '@' (Windows driver notation)
    s = s.split('@', 1)[0].strip()
    # prefer the last parentheses content if it looks like a model name
    par = re.findall(r'\(([^)]*)\)', s)
    if par:
        candidate = par[-1].strip()
        # Remove leading number prefixes like "2- " from device names
        candidate = re.sub(r'^\d+-?\s*', '', candidate)
        # Check if this looks like a proper model name (contains manufacturer/model identifiers)
        # Use parentheses content only if it has specific brand/model patterns (numbers+letters, hyphens)
        is_model_name = (
            candidate and 
            not re.search(r'^(driver|output|input|hf audio|hands-free|high definition|realtek|test|audio|device|#\d)', candidate, re.I) and
            (re.search(r'[A-Z][a-z]+\s+[A-Z]', candidate) or  # CamelCase words (Microsoft LifeChat)
             re.search(r'\w+-\w+', candidate) or  # Hyphenated model numbers (LX-3000)
             re.search(r'[A-Z]{2,}', candidate))  # Uppercase abbreviations
        )
        if is_model_name:
            base = candidate
        else:
            # Use the part before the first parenthesis
            base = s.split('(')[0].strip()
    else:
        base = s
    base = re.sub(r'[^0-9A-Za-z\-\._ ]+', ' ', base)
    base = re.sub(r'\s+', ' ', base).strip()
    return base or raw_name.strip()

def group_devices(devs):
    """
    Group devices by friendly name. Each group will contain the raw device info rows
    that look like the same physical device (different driver entries).
    """
    groups = OrderedDict()
    for d in devs:
        name = friendly_base_name(d['name'])
        key = name.lower()
        if key not in groups:
            groups[key] = {"display": name, "members": []}
        groups[key]["members"].append(d)
    # keep only groups that expose at least one input or one output channel
    filtered = OrderedDict()
    for k, g in groups.items():
        has_in = any(m['maxInputChannels'] > 0 for m in g['members'])
        has_out = any(m['maxOutputChannels'] > 0 for m in g['members'])
        if has_in or has_out:
            filtered[k] = g
    return filtered

def build_both_groups(devs):
    """
    Return only groups that have BOTH input and output indices.
    Each group entry contains:
      - display name
      - input_idxs (list)
      - output_idxs (list)
      - recommended input idx (first)
      - recommended output idx (first)
    """
    groups = group_devices(devs)
    both = []
    for k, g in groups.items():
        members = g['members']
        input_idxs = [m['index'] for m in members if m['maxInputChannels'] > 0]
        output_idxs = [m['index'] for m in members if m['maxOutputChannels'] > 0]
        if input_idxs and output_idxs:
            both.append({
                "display": g['display'],
                "input_idxs": sorted(set(input_idxs)),
                "output_idxs": sorted(set(output_idxs)),
                "rec_in": sorted(set(input_idxs))[0],
                "rec_out": sorted(set(output_idxs))[0],
            })
    return both

def pick_device_with_both():
    """
    Main function:
    - prints clean one-line-per-device (only devices that have both input and output)
    - prompts user to pick (by number) or press Enter to accept the first device
    - returns tuple: (input_index, output_index, display_name) or (None, None, None)
    """
    devs = list_devices_info()
    both_groups = build_both_groups(devs)

    if not both_groups:
        print("No devices found that have both input and output channels.")
        return None, None, None

    print("\nConnected devices (only those with BOTH input & output):\n")
    for i, g in enumerate(both_groups, start=1):
        print(f"  [{i}] {g['display']}")
        print(f"      input indices: {g['input_idxs']}    output indices: {g['output_idxs']}")
        print(f"      recommended -> input: {g['rec_in']}   output: {g['rec_out']}\n")

    prompt = ("Enter the number for the device to use (e.g. 2) and press Enter.\n"
              "Press Enter with no input to accept the first device (recommended). Your choice: ")
    raw = input(prompt).strip()

    if raw == "":
        chosen = both_groups[0]
        print(f"\nSelected (default): {chosen['display']}  -> input: {chosen['rec_in']}, output: {chosen['rec_out']}")
        return chosen['rec_in'], chosen['rec_out'], chosen['display']

    try:
        sel = int(raw)
    except ValueError:
        print("Invalid selection (not a number). Aborting selection.")
        return None, None, None

    if sel < 1 or sel > len(both_groups):
        print("Selection out of range.")
        return None, None, None

    chosen = both_groups[sel-1]
    print(f"\nSelected: {chosen['display']}  -> input: {chosen['rec_in']}, output: {chosen['rec_out']}")
    return chosen['rec_in'], chosen['rec_out'], chosen['display']


# Example usage:
if __name__ == "__main__":
    in_idx, out_idx, dev_name = pick_device_with_both()
    print("Returned indices:", in_idx, out_idx, dev_name)
