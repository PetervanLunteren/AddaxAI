"""The fixed vocabularies for what a person can record on an observation
cohort besides the count: sex, life stage and behaviour.

One place for the backend; `frontend/src/lib/observation-attributes.ts`
mirrors it for the dropdowns, the same pairing as confidence.py /
confidence.ts. The API validates against these, so the frontend copy can
never put a value in the database that this file does not know.

`sex` and `life_stage` are Camtrap DP's own enums, so they export as is.
`behavior` is free text in the standard ("preferably controlled values");
the list here is the one AddaxAI Connect uses, kept identical so the two
products' exports share a vocabulary. American spelling on the wire and
in the database, matching the standard's column name.

NULL means unknown everywhere. Never store a literal "unknown": it is not
in the Camtrap DP enums and every query would have to special-case it.
"""

from typing import Literal

SEXES = ("female", "male")
LIFE_STAGES = ("adult", "subadult", "juvenile")
BEHAVIORS = (
    "traveling",
    "foraging",
    "resting",
    "vigilance",
    "drinking",
    "grooming",
    "courtship",
    "nursing",
    "aggression",
    "marking",
)

Sex = Literal["female", "male"]
LifeStage = Literal["adult", "subadult", "juvenile"]
Behavior = Literal[
    "traveling",
    "foraging",
    "resting",
    "vigilance",
    "drinking",
    "grooming",
    "courtship",
    "nursing",
    "aggression",
    "marking",
]
