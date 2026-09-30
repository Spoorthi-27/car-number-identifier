"""Plate normalization and Karnataka RTO lookup. No OCR or camera required."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.district import UNKNOWN_DISTRICT, get_district, lookup_registration, normalize_plate


def test_format_variants_normalize_to_the_same_plate():
    assert normalize_plate("KA 09 AB 1234") == "KA09AB1234"
    assert normalize_plate("KA-09-AB-1234") == "KA09AB1234"
    assert normalize_plate("KA09AB1234") == "KA09AB1234"
    assert normalize_plate("ka-09-ab-1234") == "KA09AB1234"


def test_ambiguous_characters_change_only_in_the_matching_slot():
    assert normalize_plate("KAO9AB1234") == "KA09AB1234"
    assert normalize_plate("KA09AB123S") == "KA09AB1235"
    assert normalize_plate("KA09A81234") == "KA09AB1234"
    # A real series letter is not rewritten into a digit.
    assert normalize_plate("KA09ABI234") == "KA09ABI234"
    # An unambiguous digit is not rewritten into a letter.
    assert normalize_plate("KA09AB1235") == "KA09AB1235"


def test_ka09_is_mysuru_for_every_format():
    for raw in ("KA09AB1234", "KA-09-AB-1234", "KA 09 AB 1234", "KA 09", "KA-09", "KA09"):
        info = lookup_registration(raw)
        assert info.state == "Karnataka"
        assert info.state_code == "KA"
        assert info.rto_code == "KA09"
        assert info.district == "Mysuru"
        assert get_district(raw) == "Mysuru"


def test_another_rto_code_resolves_its_own_district():
    info = lookup_registration("KA11CD5678")
    assert info.rto_code == "KA11"
    assert info.district == "Mandya"
    assert info.rto_name == "Mandya"

    mangaluru = lookup_registration("KA-19-MN-4321")
    assert mangaluru.rto_code == "KA19"
    assert mangaluru.district == "Dakshina Kannada"
    assert mangaluru.rto_name == "Mangaluru"


def test_one_stray_ocr_character_is_removed_once():
    assert normalize_plate("KAQO9AB1234") == "KA09AB1234"
    assert normalize_plate("KAI99ZZ9999") == "KA99ZZ9999"
    # A long failed read must not be rewritten into a different real RTO.
    assert lookup_registration("KAI99ZZ9999").district == UNKNOWN_DISTRICT
    assert lookup_registration("KAI99ZZ9999").rto_code == "KA99"


def test_unknown_rto_does_not_crash():
    info = lookup_registration("KA99AB1234")
    assert info.state == "Karnataka"
    assert info.rto_code == "KA99"
    assert info.district == UNKNOWN_DISTRICT

    assert lookup_registration("MH12AB1234").state == "Maharashtra"
    assert lookup_registration("MH12AB1234").district == UNKNOWN_DISTRICT
    assert lookup_registration("").district == UNKNOWN_DISTRICT
    assert lookup_registration(None).district == UNKNOWN_DISTRICT
    assert lookup_registration("not a plate").district == UNKNOWN_DISTRICT
