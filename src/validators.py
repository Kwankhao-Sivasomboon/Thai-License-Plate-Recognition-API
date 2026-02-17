from pydantic import BaseModel, field_validator, ValidationError
import re
from typing import Optional

# Thai Consonants Range: \u0E01-\u0E2E (Ko Kai to Ho Nokhuk)
# Note: Some specialized plates might use other chars, but user said "NCC", "CC" (Char) which implies consonants.
# User's "N" = Number (Digit), "C" = Character (Thai Consonant).

THAI_CONSONANTS = r"[\u0E01-\u0E2E]"
DIGIT = r"\d"

# Regex Patterns
PATTERN_NCC_NNNN = re.compile(rf"^{DIGIT}{THAI_CONSONANTS}{{2}} {DIGIT}{{1,4}}$") # e.g., 1กข 1234
PATTERN_CC_NNNN  = re.compile(rf"^{THAI_CONSONANTS}{{2}} {DIGIT}{{1,4}}$")         # e.g., กข 1234
PATTERN_NN_NNNN  = re.compile(rf"^{DIGIT}{{2}}-{DIGIT}{{4}}$")                   # e.g., 70-1234

class PlateLabelValidator(BaseModel):
    text: str
    
    @field_validator('text')
    @classmethod
    def validate_format(cls, v: str) -> str:
        # Check against the 3 allowed patterns
        is_ncc = PATTERN_NCC_NNNN.match(v)
        is_cc  = PATTERN_CC_NNNN.match(v)
        is_nn  = PATTERN_NN_NNNN.match(v)
        
        if not (is_ncc or is_cc or is_nn):
            raise ValueError(f"Invalid plate format: '{v}'. Must match NCC NNNN, CC NNNN, or NN-NNNN.")
        
        return v

def is_valid_plate(upload_text: str) -> bool:
    """Returns True if the text matches strict plate rules."""
    try:
        PlateLabelValidator(text=upload_text)
        return True
    except ValidationError:
        return False
