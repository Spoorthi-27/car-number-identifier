"""
district.py
------------
Maps the RTO code embedded in a Karnataka plate (e.g. "KA09...") to the
district/area name it was registered in, so the logs can show where a
detected car is from, not just its raw plate text.

Indian plates follow STATE(2 letters) + RTO CODE(2 digits) + series + number,
e.g. "KA09AB1234" -> state "KA", RTO code "09" -> Mysuru West.

To support another state, add a similarly-shaped dict below and register it
in STATE_RTO_CODES.
"""
import re
from typing import Optional

KARNATAKA_RTO_CODES = {
    "01": "Bengaluru Central",
    "02": "Bengaluru West",
    "03": "Bengaluru East",
    "04": "Bengaluru North",
    "05": "Bengaluru South",
    "06": "Tumakuru",
    "07": "Kolar",
    "08": "KGF",
    "09": "Mysuru West",
    "10": "Chamarajanagar",
    "11": "Mandya",
    "12": "Madikeri / Kodagu",
    "13": "Hassan",
    "14": "Shivamogga",
    "15": "Sagara",
    "16": "Chitradurga",
    "17": "Davanagere",
    "18": "Chikkamagaluru",
    "19": "Mangaluru",
    "20": "Udupi",
    "21": "Puttur",
    "22": "Belagavi",
    "23": "Chikkodi",
    "24": "Bailhongal",
    "25": "Dharwad",
    "26": "Gadag",
    "27": "Haveri",
    "28": "Vijayapura",
    "29": "Bagalkot",
    "30": "Karwar",
    "31": "Sirsi",
    "32": "Kalaburagi",
    "33": "Yadgir",
    "34": "Ballari",
    "35": "Hosapete",
    "36": "Raichur",
    "37": "Koppal",
    "38": "Bidar",
    "39": "Bhalki",
    "40": "Chikkaballapur",
    "41": "Bengaluru/Jnanabharathi",
    "42": "Ramanagara",
    "43": "Devanahalli",
    "44": "Tiptur",
    "45": "Hunsur",
    "46": "Sakleshpur",
    "47": "Honnavar",
    "48": "Jamkhandi",
    "49": "Gokak",
    "50": "Yelahanka",
    "51": "Electronic City",
    "52": "Nelamangala",
    "53": "K.R. Puram",
    "54": "Nagamangala",
    "55": "Mysuru East",
    "56": "Basavakalyan",
    "57": "Shantinagar, Bengaluru",
    "59": "Chandapura",
    "61": "Marathahalli",
    "62": "Surathkal",
    "63": "Dharwad East / Hubballi",
    "64": "Madhugiri",
    "65": "Dandeli",
    "66": "Tarikere",
    "67": "Sedam",
    "68": "Chintamani",
    "69": "Ranebennuru",
    "70": "Bantwal",
}

# Registry of state code -> RTO code dict. Add more states here as needed.
STATE_RTO_CODES = {
    "KA": KARNATAKA_RTO_CODES,
}

# A full Indian plate shape: STATE(2 letters) + RTO(2 digits) + series(1-3
# letters) + number(4 digits), e.g. "KA70JK7890". Restricted to KNOWN state
# codes (not just any 2 letters) so noise elsewhere in the OCR text can't
# accidentally look like a different, unsupported state. Searched anywhere
# in the cleaned OCR text (not just anchored at position 0), since OCR
# commonly picks up one stray leading character - typically from the small
# "IND" hologram chip printed next to the plate text - that an anchored
# match would be thrown off by.
_KNOWN_STATE_PATTERN = "|".join(sorted(STATE_RTO_CODES.keys()))
_CLEAN_PLATE_RE = re.compile(
    rf"(?:{_KNOWN_STATE_PATTERN})\d{{2}}[A-Z]{{1,3}}\d{{4}}"
)
_STATE_RTO_RE = re.compile(rf"({_KNOWN_STATE_PATTERN})(\d{{2}})")

# Generic plate shape (ANY 2-letter state, not just known ones), anchored to
# the whole string. Used as a fallback so plates from states we don't have
# an RTO table for can still be recognized and logged - but only when the
# OCR text is already completely clean, with no stray characters around it,
# since we can't cross-check an arbitrary state code against a known list
# the way we can for _CLEAN_PLATE_RE above.
_GENERIC_PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}$")

# Loose fallback used only for the district *hint*: any 2 letters + 2 digits
# at the very start, for reads that aren't a full clean plate but might
# still carry a genuine (if unverified) state+RTO prefix.
_PLATE_PREFIX_RE = re.compile(r"^([A-Z]{2})(\d{2})")


def extract_valid_plate(raw_text: str) -> Optional[str]:
    """
    Try to recover a genuine, complete plate reading from noisy OCR text,
    e.g. "BKA70JK7890" (stray leading "B") -> "KA70JK7890", or
    "EKA01AB1234" (stray leading "E") -> "KA01AB1234".

    For a KNOWN state (see STATE_RTO_CODES), this searches anywhere in the
    text and tolerates stray characters around the real plate. For any
    other state, it only accepts the text if it's already an exact, clean
    plate shape with nothing extra around it - we can't safely strip noise
    around a state code we don't have a reference list for.

    Reads missing a whole segment (a dropped state prefix, or a dropped
    trailing digit group) correctly return None rather than a partial or
    wrong guess.
    """
    if not raw_text:
        return None

    text = raw_text.upper().strip()
    match = _CLEAN_PLATE_RE.search(text)
    if match:
        return match.group(0)

    return text if _GENERIC_PLATE_RE.match(text) else None


def get_district(plate_text: str) -> Optional[str]:
    """
    Given a cleaned plate string (e.g. "KA09AB1234", or a noisy OCR read
    like "BKA70JK7890"), return the district/area name for its RTO code, or
    None if the state isn't supported or no recognizable STATE+RTO prefix
    can be found.
    """
    if not plate_text:
        return None

    text = plate_text.upper()
    match = _STATE_RTO_RE.search(text) or _PLATE_PREFIX_RE.match(text)
    if not match:
        return None

    state_code, rto_code = match.groups()
    rto_table = STATE_RTO_CODES.get(state_code)
    if rto_table is None:
        return None

    return rto_table.get(rto_code)
