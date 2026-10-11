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


def test_lookalikes_are_other_genera_of_the_same_kingdom():
    lookalikes = json.loads((DATA / "lookalikes.json").read_text())
    assert len(lookalikes) == len(species)
    for i, near in enumerate(lookalikes):
        assert all(species[j]["kingdom"] == species[i]["kingdom"] for j in near)
        assert all(species[j]["genus"] != species[i]["genus"] for j in near)


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


def test_classic_deadly_mixups_raise_the_alarm():
    """If the top match is one of these, the app must warn about the deadly look-alike (via lookalikes.json)."""
    lookalikes = json.loads((DATA / "lookalikes.json").read_text())
    index = {s["name"]: i for i, s in enumerate(species)}
    cases = {
        "Daucus carota": "Conium",           # wild carrot -> poison hemlock
        "Coriandrum sativum": "Conium",      # coriander -> poison hemlock
        "Morchella esculenta": "Gyromitra",  # morel -> false morel
        "Allium ursinum": None,              # wild garlic -> lily of the valley / autumn crocus / arum
    }
    for name, genus in cases.items():
        near = {species[j]["genus"] for j in lookalikes[index[name]]}
        flagged = near & set(danger["genera"])
        assert flagged if genus is None else genus in flagged, f"{name}: no warning ({sorted(near)})"


def test_known_lookalikes_point_at_flagged_genera():
    for genus, dangerous in danger["known_lookalikes"].items():
        assert genus not in danger["genera"], f"{genus} is itself toxic; list it under genera instead"
        for d in dangerous:
            assert d in danger["genera"], f"{genus} -> {d}: {d} has no warning text"
    assert "Omphalotus" in danger["known_lookalikes"]["Cantharellus"]
