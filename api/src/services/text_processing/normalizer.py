import re

def split_num(num: re.Match) -> str:
    """Handle number splitting for various formats"""
    num = num.group()
    if "." in num:
        return num
    elif ":" in num:
        h, m = [int(n) for n in num.split(":")]
        if m == 0:
            return f"{h} o'clock"
        elif m < 10:
            return f"{h} oh {m}"
        return f"{h} {m}"
    year = int(num[:4])
    if year < 1100 or year % 1000 < 10:
        return num
    left, right = num[:2], int(num[2:4])
    s = "s" if num.endswith("s") else ""
    if 100 <= year % 1000 <= 999:
        if right == 0:
            return f"{left} hundred{s}"
        elif right < 10:
            return f"{left} oh {right}{s}"
    return f"{left} {right}{s}"

def handle_money(m: re.Match) -> str:
    """Convert money expressions to spoken form"""
    m = m.group()
    bill = "dollar" if m[0] == "$" else "pound"
    if m[-1].isalpha():
        return f"{m[1:]} {bill}s"
    elif "." not in m:
        s = "" if m[1:] == "1" else "s"
        return f"{m[1:]} {bill}{s}"
    b, c = m[1:].split(".")
    s = "" if b == "1" else "s"
    c = int(c.ljust(2, "0"))
    coins = (
        f"cent{'' if c == 1 else 's'}"
        if m[0] == "$"
        else ("penny" if c == 1 else "pence")
    )
    return f"{b} {bill}{s} and {c} {coins}"

def handle_decimal(num: re.Match) -> str:
    """Convert decimal numbers to spoken form"""
    a, b = num.group().split(".")
    return " point ".join([a, " ".join(b)])

def normalize_text(text: str) -> str:
    """Normalize text for TTS processing
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    # Replace quotes and brackets
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace("«", chr(8220)).replace("»", chr(8221))
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    text = text.replace("(", "«").replace(")", "»")
    
    # Handle CJK punctuation
    for a, b in zip("、。！，：；？", ",.!,:;?"):
        text = text.replace(a, b + " ")
    
    # Clean up whitespace
    text = re.sub(r"[^\S \n]", " ", text)
    text = re.sub(r"  +", " ", text)
    text = re.sub(r"(?<=\n) +(?=\n)", "", text)
    
    # Handle titles and abbreviations
    text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
    text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
    text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
    text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
    text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)
    
    # Handle common words
    text = re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)
    
    # Handle numbers and money
    text = re.sub(
        r"\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)", 
        split_num, 
        text
    )
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(
        r"(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b",
        handle_money,
        text,
    )
    text = re.sub(r"\d*\.\d+", handle_decimal, text)
    
    # Handle various formatting
    text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
    text = re.sub(r"(?<=\d)S", " S", text)
    text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
    text = re.sub(r"(?<=X')S\b", "s", text)
    text = re.sub(
        r"(?:[A-Za-z]\.){2,} [a-z]", 
        lambda m: m.group().replace(".", "-"), 
        text
    )
    text = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", text)
    
    return text.strip()
