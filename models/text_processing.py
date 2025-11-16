import difflib
import re

CANONICAL_COMMANDS = [
    "create note",
    "delete note",
    "list notes",
    "send email",
    "check gmail",
    "create event",
    "list events",
    "search event",
    "exit",
    # add more commands you support, exact canonical phrases you want to force
]

def _clean_text_for_compare(s: str) -> str:
    s = s.lower().strip()
    # remove punctuation except keep spaces and alphanumerics
    s = re.sub(r'[^0-9a-z\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# --- helper: find best matching canonical command using only the first N words ---
def find_best_command_match(transcript: str, commands=CANONICAL_COMMANDS):
    """
    Returns a tuple (best_command, similarity_float, command_word_count)
    or (None, 0.0, 0) if no good candidates.
    similarity is in [0.0, 1.0].
    """
    if not transcript or transcript.strip() == "":
        return None, 0.0, 0

    cleaned = _clean_text_for_compare(transcript)
    words = cleaned.split()
    if not words:
        return None, 0.0, 0

    best = (None, 0.0, 0)
    for cmd in commands:
        cmd_clean = _clean_text_for_compare(cmd)
        cmd_words = cmd_clean.split()
        if not cmd_words:
            continue
        n = len(cmd_words)
        # if transcript shorter than n words, still compare available portion
        prefix = " ".join(words[:n])
        # compute similarity ratio
        ratio = difflib.SequenceMatcher(None, prefix, cmd_clean).ratio()
        if ratio > best[1]:
            best = (cmd, ratio, n)
    return best  # (command, ratio, n)

# --- interactive disambiguation wrapper (call after you transcribe) ---
def confirm_and_apply_command_correction(transcript: str, threshold: float = 0.90):
    """
    If the start of `transcript` is similar to a canonical command >= threshold,
    prompt the user: "Did you mean '...'? (y/n)". If yes, replace the first N words
    in the transcript with the canonical command and return the new transcript.
    Otherwise return original transcript.
    """
    cmd, similarity, n = find_best_command_match(transcript)
    if cmd is None:
        return transcript

    if similarity < threshold or similarity > 0.98:
        return transcript
    # Prompt user (text prompt). Use full human-readable candidate.
    print(f"\nDid you mean the command: '{cmd}' ?  (similarity {similarity*100:.1f}%)")

    ans = input("Type 'y' or 'yes' to accept, anything else to keep original: ").strip().lower()
    if ans in ("y", "yes"):
        # replace first n words of original transcript (preserve rest)
        orig_words = transcript.split()
        # if original transcript has fewer than n words, just use command alone
        rest = orig_words[n:] if len(orig_words) > n else []
        new_transcript = " ".join([cmd] + rest).strip()
        print(f"[AUTOCORRECT] Using clarified command: {new_transcript}")
        return new_transcript
    else:
        print("[AUTOCORRECT] Keeping original transcript.")
        return transcript