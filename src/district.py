"""
district.py
-----------
Normalize Indian registration numbers and resolve Karnataka RTO codes to a
district.

Plate text is accepted in any of the common shapes ("KA09AB1234",
"KA-09-AB-1234", "KA 09 AB 1234", "KA 09"). The value stored and shown is
the compact form, for example KA09AB1234.

Character fixes are applied only to the ambiguous pairs O/0, I/1, B/8, S/5
and Z/2, and only after a strict parse (no character changes) has failed.
An unambiguous character is never rewritten. When a strict parse already
fits the plate, that reading is kept as-is.

Unknown state or RTO codes return district "Unknown RTO/District" and do
not raise.
"""
import re
from dataclasses import dataclass
from typing import Optional

UNKNOWN_DISTRICT = "Unknown RTO/District"

# Ambiguous glyphs. Used only in the fallback parse, and only in the slot
# (letter or digit) where that glyph is illegal without a substitution.
_TO_LETTER = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z", "6": "G"}
_TO_DIGIT = {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2", "G": "6"}

# code -> (RTO office name, district)
# Office names follow the Karnataka Transport Department list. The district
# is what the API, logs, and UI show. KA09 is Mysuru, not "Mysuru West".
_KARNATAKA_RTO_DETAILS = {
    "01": ("Bengaluru Central", "Bengaluru"),
    "02": ("Bengaluru West", "Bengaluru"),
    "03": ("Bengaluru East", "Bengaluru"),
    "04": ("Bengaluru North", "Bengaluru"),
    "05": ("Bengaluru South", "Bengaluru"),
    "06": ("Tumakuru", "Tumakuru"),
    "07": ("Kolar", "Kolar"),
    "08": ("KGF", "Kolar"),
    "09": ("Mysuru West", "Mysuru"),
    "10": ("Chamarajanagar", "Chamarajanagar"),
    "11": ("Mandya", "Mandya"),
    "12": ("Madikeri", "Kodagu"),
    "13": ("Hassan", "Hassan"),
    "14": ("Shivamogga", "Shivamogga"),
    "15": ("Sagara", "Shivamogga"),
    "16": ("Chitradurga", "Chitradurga"),
    "17": ("Davanagere", "Davanagere"),
    "18": ("Chikkamagaluru", "Chikkamagaluru"),
    "19": ("Mangaluru", "Dakshina Kannada"),
    "20": ("Udupi", "Udupi"),
    "21": ("Puttur", "Dakshina Kannada"),
    "22": ("Belagavi", "Belagavi"),
    "23": ("Chikkodi", "Belagavi"),
    "24": ("Bailhongal", "Belagavi"),
    "25": ("Dharwad", "Dharwad"),
    "26": ("Gadag", "Gadag"),
    "27": ("Haveri", "Haveri"),
    "28": ("Vijayapura", "Vijayapura"),
    "29": ("Bagalkot", "Bagalkot"),
    "30": ("Karwar", "Uttara Kannada"),
    "31": ("Sirsi", "Uttara Kannada"),
    "32": ("Kalaburagi", "Kalaburagi"),
    "33": ("Yadgir", "Yadgir"),
    "34": ("Ballari", "Ballari"),
    "35": ("Hosapete", "Vijayanagara"),
    "36": ("Raichur", "Raichur"),
    "37": ("Koppal", "Koppal"),
    "38": ("Bidar", "Bidar"),
    "39": ("Bhalki", "Bidar"),
    "40": ("Chikkaballapur", "Chikkaballapur"),
    "41": ("Jnanabharathi", "Bengaluru"),
    "42": ("Ramanagara", "Ramanagara"),
    "43": ("Devanahalli", "Bengaluru Rural"),
    "44": ("Tiptur", "Tumakuru"),
    "45": ("Hunsur", "Mysuru"),
    "46": ("Sakleshpur", "Hassan"),
    "47": ("Honnavar", "Uttara Kannada"),
    "48": ("Jamkhandi", "Bagalkot"),
    "49": ("Gokak", "Belagavi"),
    "50": ("Yelahanka", "Bengaluru"),
    "51": ("Electronic City", "Bengaluru"),
    "52": ("Nelamangala", "Bengaluru Rural"),
    "53": ("K.R. Puram", "Bengaluru"),
    "54": ("Nagamangala", "Mandya"),
    "55": ("Mysuru East", "Mysuru"),
    "56": ("Basavakalyan", "Bidar"),
    "57": ("Shantinagar", "Bengaluru"),
    "58": ("Banashankari", "Bengaluru"),
    "59": ("Chandapura", "Bengaluru"),
    "60": ("R.T. Nagar", "Bengaluru"),
    "61": ("Marathahalli", "Bengaluru"),
    "62": ("Surathkal", "Dakshina Kannada"),
    "63": ("Hubballi", "Dharwad"),
    "64": ("Madhugiri", "Tumakuru"),
    "65": ("Dandeli", "Uttara Kannada"),
    "66": ("Tarikere", "Chikkamagaluru"),
    "67": ("Sedam", "Kalaburagi"),
    "68": ("Chintamani", "Chikkaballapur"),
    "69": ("Ranebennuru", "Haveri"),
    "70": ("Bantwal", "Dakshina Kannada"),
    "71": ("Athani", "Belagavi"),
}

# Backward-compatible view: RTO code -> district name.
KARNATAKA_RTO_CODES = {code: district for code, (_office, district) in _KARNATAKA_RTO_DETAILS.items()}

STATE_RTO_CODES = {
    "KA": KARNATAKA_RTO_CODES,
}

STATE_NAMES = {
    "AN": "Andaman and Nicobar",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

_BH_RE = re.compile(r"\d{2}BH\d{4}[A-Z]{1,2}")
_MAX_STRAY_CHARS = 3
_MAX_SUBSTITUTIONS = 3


@dataclass
class RegistrationInfo:
    """Resolved plate identity. `district` is never None."""

    raw_text: str
    normalized: Optional[str]
    state: Optional[str]
    state_code: Optional[str]
    rto_code: Optional[str]
    rto_name: Optional[str]
    district: str

    @classmethod
    def unknown(cls, raw_text: str = "") -> "RegistrationInfo":
        return cls(
            raw_text=raw_text or "",
            normalized=None,
            state=None,
            state_code=None,
            rto_code=None,
            rto_name=None,
            district=UNKNOWN_DISTRICT,
        )


@dataclass
class _Parsed:
    plate: str
    consumed: int
    substitutions: int
    state: str
    rto: str
    series_len: int
    number_len: int
    padded_rto: bool = False


def _clean_alnum(raw: Optional[str]) -> str:
    if not raw:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(raw).upper())


def _take(token: str, kind: str, allow_ambiguous: bool) -> tuple:
    """
    Map `token` into letters or digits.

    Strict mode accepts only characters that already belong in the slot.
    Ambiguous mode may rewrite O/0, I/1, B/8, S/5, Z/2. Any other mismatch
    rejects the token instead of guessing.
    """
    out = []
    substitutions = 0
    for char in token:
        if kind == "letter":
            if char.isalpha():
                out.append(char)
                continue
            if allow_ambiguous and char in _TO_LETTER:
                out.append(_TO_LETTER[char])
                substitutions += 1
                continue
            return None, 0
        if char.isdigit():
            out.append(char)
            continue
        if allow_ambiguous and char in _TO_DIGIT:
            out.append(_TO_DIGIT[char])
            substitutions += 1
            continue
        return None, 0
    return "".join(out), substitutions


def _split_tail(tail: str, allow_ambiguous: bool) -> list:
    """Every legal series/number split. The caller ranks them."""
    options = []
    for series_len in (1, 2, 3):
        for number_len in (1, 2, 3, 4):
            need = series_len + number_len
            if need > len(tail):
                continue
            series, series_subs = _take(tail[:series_len], "letter", allow_ambiguous)
            if series is None:
                continue
            number, number_subs = _take(tail[series_len:need], "digit", allow_ambiguous)
            if number is None:
                continue
            options.append((series, number, series_subs + number_subs, need, series_len, number_len))
    return options


def _parse_prefix(text: str, allow_ambiguous: bool) -> list:
    """Every plate that can start at text[0]. Does not look at later windows."""
    if len(text) < 5:
        return []
    state, state_subs = _take(text[:2], "letter", allow_ambiguous)
    if not state:
        return []

    rest = text[2:]
    found = []

    def consider(rto: str, rto_subs: int, rto_consumed: int, tail: str, padded_rto: bool) -> None:
        for series, number, tail_subs, tail_consumed, series_len, number_len in _split_tail(tail, allow_ambiguous):
            found.append(
                _Parsed(
                    plate=f"{state}{rto}{series}{number}",
                    consumed=2 + rto_consumed + tail_consumed,
                    substitutions=state_subs + rto_subs + tail_subs,
                    state=state,
                    rto=rto,
                    series_len=series_len,
                    number_len=number_len,
                    padded_rto=padded_rto,
                )
            )

    two_digit = []
    if len(rest) >= 3:
        rto, rto_subs = _take(rest[:2], "digit", allow_ambiguous)
        if rto is not None:
            consider(rto, rto_subs, 2, rest[2:], padded_rto=False)
            two_digit = found[:]

    # "KA9AB1234" has a one-digit RTO. Pad it only when two digits do not fit,
    # so "KA91..." is not rewritten as "KA09".
    if not two_digit and rest:
        one, one_subs = _take(rest[:1], "digit", allow_ambiguous)
        second_is_digit = len(rest) > 1 and _take(rest[1:2], "digit", allow_ambiguous)[0] is not None
        if one is not None and not second_is_digit:
            consider(f"0{one}", one_subs, 1, rest[1:], padded_rto=True)

    return found


def _try_bh_series(text: str) -> Optional[str]:
    """Bharat series, for example 22BH1234AB. Accepted only as an exact token."""
    match = _BH_RE.search(text)
    if not match:
        return None
    token = match.group(0)
    stray = len(text) - len(token)
    if stray > _MAX_STRAY_CHARS:
        return None
    return token


def _rank_key(parsed: _Parsed, unconsumed: int) -> tuple:
    known_state = parsed.state in STATE_NAMES
    # Padding a 1-digit RTO invents a zero. That costs more than one
    # ambiguous-slot fix, so "KA0S..." becomes KA05 rather than KA00S....
    # A genuine "KA9AB1234" still pads, because nothing else parses.
    edit_cost = parsed.substitutions + (2 if parsed.padded_rto else 0)
    return (
        unconsumed,
        edit_cost,
        0 if known_state else 1,
        0 if parsed.number_len == 4 else 1,
        0 if parsed.series_len == 2 else 1,
        -len(parsed.plate),
    )


def _best_parse(text: str) -> tuple:
    """Best plate parse, plus how many source characters it did not use."""
    best = None
    best_key = None
    best_unconsumed = 0
    # Using every character beats a shorter strict read that drops a trailing
    # ambiguous glyph, but a zero-substitution read still beats a rewrite when
    # both consume the same text. That keeps KA09ABI234 intact.
    for allow_ambiguous in (False, True):
        for start in range(len(text)):
            for parsed in _parse_prefix(text[start:], allow_ambiguous):
                if parsed.substitutions > _MAX_SUBSTITUTIONS:
                    continue
                unconsumed = start + (len(text[start:]) - parsed.consumed)
                known_state = parsed.state in STATE_NAMES
                if not known_state and unconsumed > 0:
                    continue
                if known_state and unconsumed > _MAX_STRAY_CHARS:
                    continue
                key = _rank_key(parsed, unconsumed)
                if best_key is None or key < best_key:
                    best_key = key
                    best = parsed
                    best_unconsumed = unconsumed
    return best, best_unconsumed


def _recover_single_deletion(text: str) -> Optional[str]:
    """
    Drop one stray OCR character when that produces exactly one plate.

    "KAQO9AB1234" (an extra Q) becomes KA09AB1234. If two different deletions
    produce two different plates, this returns None instead of guessing.
    """
    if not (6 <= len(text) <= 16):
        return None
    found = {}
    for index in range(len(text)):
        shortened = text[:index] + text[index + 1:]
        parsed, unconsumed = _best_parse(shortened)
        if parsed is None or unconsumed != 0 or parsed.state not in STATE_NAMES:
            continue
        if parsed.substitutions > 2:
            continue
        found.setdefault(parsed.plate, _rank_key(parsed, 0))
    if not found:
        return None
    best_key = min(found.values())
    winners = [plate for plate, key in found.items() if key == best_key]
    if len(winners) != 1:
        return None
    return winners[0]


def normalize_plate(raw: Optional[str]) -> Optional[str]:
    """
    Return a compact registration number, or None when the text is not a plate.

    Examples:
        "KA 09 AB 1234" -> "KA09AB1234"
        "KA-09-AB-1234" -> "KA09AB1234"
        "KAO9AB1234"    -> "KA09AB1234"  (O was in the RTO digit slot)
        "KA09ABI234"    -> "KA09ABI234"  (I stays; it is a real series letter)
    """
    text = _clean_alnum(raw)
    if not text:
        return None

    parsed, _unconsumed = _best_parse(text)
    if parsed is not None:
        return parsed.plate

    recovered = _recover_single_deletion(text)
    if recovered:
        return recovered
    return _try_bh_series(text)


def extract_valid_plate(raw_text: str) -> Optional[str]:
    """Recover a plate from noisy OCR text. None when it cannot be done safely."""
    return normalize_plate(raw_text)


def _prefix_state_rto(raw: Optional[str]) -> tuple:
    """
    Read a state + RTO even when the rest of the plate is missing.

    "KA-09" and "KA 09" resolve to ("KA", "09") so the district can still
    be shown for a partial but genuine prefix.
    """
    normalized = normalize_plate(raw)
    if normalized and not normalized[2:4].isdigit():
        normalized = None
    if normalized and len(normalized) >= 4 and normalized[:2].isalpha() and normalized[2:4].isdigit():
        if not _BH_RE.fullmatch(normalized):
            return normalized[:2], normalized[2:4]

    cleaned = _clean_alnum(raw)
    if len(cleaned) < 3:
        return None, None

    # A long string that is not a valid plate must not have its RTO guessed
    # by rewriting letters into digits. "KAI99..." is not KA19. Short fragments
    # such as "KAO9" or "KA-09" still allow the ambiguous digit fixes.
    allow_fixes = len(cleaned) <= 6
    state, _subs = _take(cleaned[:2], "letter", allow_fixes)
    if not state or not state.isalpha():
        return None, None
    rto, _rto_subs = _take(cleaned[2:4], "digit", allow_fixes)
    if rto and len(rto) == 2:
        return state, rto
    one, _one_subs = _take(cleaned[2:3], "digit", allow_fixes)
    second_is_digit = len(cleaned) > 3 and _take(cleaned[3:4], "digit", allow_fixes)[0] is not None
    if one and not second_is_digit:
        return state, f"0{one}"
    return None, None


def lookup_registration(plate_text: Optional[str]) -> RegistrationInfo:
    """
    Resolve state, RTO code, office, and district for a plate string.

    Always returns a RegistrationInfo. The district is "Unknown RTO/District"
    when the code is missing or not in the Karnataka table.
    """
    raw_text = "" if plate_text is None else str(plate_text)
    try:
        normalized = normalize_plate(raw_text)
        if normalized and _BH_RE.fullmatch(normalized):
            return RegistrationInfo(
                raw_text=raw_text,
                normalized=normalized,
                state="India",
                state_code="BH",
                rto_code=None,
                rto_name=None,
                district=UNKNOWN_DISTRICT,
            )

        state_code, rto_digits = _prefix_state_rto(normalized or raw_text)
        if not state_code:
            return RegistrationInfo(
                raw_text=raw_text,
                normalized=normalized,
                state=None,
                state_code=None,
                rto_code=None,
                rto_name=None,
                district=UNKNOWN_DISTRICT,
            )

        rto_code = f"{state_code}{rto_digits}"
        office = None
        district = UNKNOWN_DISTRICT
        table = STATE_RTO_CODES.get(state_code)
        if table and rto_digits in _KARNATAKA_RTO_DETAILS:
            office, district = _KARNATAKA_RTO_DETAILS[rto_digits]
        elif table and rto_digits in table:
            district = table[rto_digits]

        return RegistrationInfo(
            raw_text=raw_text,
            normalized=normalized,
            state=STATE_NAMES.get(state_code),
            state_code=state_code,
            rto_code=rto_code,
            rto_name=office,
            district=district,
        )
    except Exception:
        return RegistrationInfo.unknown(raw_text)


def get_district(plate_text: str) -> str:
    """District name for a plate, or "Unknown RTO/District"."""
    return lookup_registration(plate_text).district
