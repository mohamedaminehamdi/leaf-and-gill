"""Offline checks on the shipped data pack: alignment, safety coverage, no edibility claims."""
import json
import re
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent.parent / "app" / "data"
species = json.loads((DATA / "species.json").read_text())
meta = json.loads((DATA / "meta.json").read_text())
danger = json.loads((DATA / "danger.json").read_text())


def test_embeddings_line_up_with_species():
    emb = np.fromfile(DATA / "text_emb.f16", dtype="<f2").reshape(-1, meta["dim"])
    assert len(emb) == len(species) == meta["count"]
    assert np.allclose(np.linalg.norm(emb.astype(np.float32), axis=1), 1, atol=1e-2)


def test_lookalikes_are_other_species_of_the_same_kingdom():
    lookalikes = json.loads((DATA / "lookalikes.json").read_text())
    assert len(lookalikes) == len(species)
    for i, near in enumerate(lookalikes):
        assert i not in near
        assert all(species[j]["kingdom"] == species[i]["kingdom"] for j in near)


def test_every_deadly_species_is_in_the_pack_and_flagged():
    names = {s["name"]: s for s in species}
    for name in danger["must_include"]:
        assert name in names, f"{name} missing: the app could never warn about it"
        assert names[name]["genus"] in danger["genera"], f"{name}'s genus has no warning text"


def test_no_edibility_claims_anywhere():
    claim = re.compile(r"\b(edible|safe to eat|tasty|delicious|good to eat|choice edible)\b", re.I)
    for path in DATA.glob("*.json"):
        text = path.read_text()
        assert not claim.search(text), f"{path.name} mentions edibility: {claim.search(text).group(0)}"
