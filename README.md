# Paduka

A prosthetic limb designed to be shown, not hidden.

A below-knee prosthesis in six printed pieces, standing on an ornamented
*paduka* — the carved sole-and-toe-knob sandal that is one of the oldest
forms of footwear in India, and which never once pretends to be a bare foot.
Block-printed ornament, a working single-axis ankle, and a wood-fibre
biopolymer, on a printer that fits on a bench.

**→ [The site](https://vivek-kommi.github.io/paduka/)**

Submitted to the [British Cræft Prize](https://www.nationofartisans.com/prize), 2026.

---

## What is actually here

Three things, and the interesting one is the second.

**A printable outer form.** Cuff, shank, ankle, foot, paduka. Six pieces,
420 mm, about 2.3 litres of material, assembling without tools.

**A system that generates the ornament.** Not a picture wrapped round a
standard foot: the drawing is laid out in an unrolled (u, v) chart, cut into
separate blocks the way a Sanganeri print is cut, and turned into real
geometry — each dye a recess in the ground with a plug that fills it flush.
Six traditions are implemented, and each one is worked in zones up the leg
rather than tiled flat, because that is how the cloth it comes from is made
up.

**A joint.** The foot parts at the malleoli into a single-axis ankle: a 6 mm
pin in double shear, about ±9° of travel, and a bumper either side of the pin
doing the tuning.

## Why it is built the way it is

On a multi-material printer, colour is **not** charged per motif. It is
charged **per layer**. Any layer containing two filaments costs one tool
change and one purge, whether it carries a whole paisley or a single dot. A
420 mm object at 0.2 mm is about 2,100 layers, so an all-over repeat is on the
order of 1.9 kg of purge for a 2.1 kg object.

That single fact decides nearly everything about the printed drawing: the
ornament is gathered into worked borders with plain single-filament ground
between them, motifs are laid out by *count* rather than pitch so nothing is
ever sliced at the seam, and nothing is printed in an ink colour where it
cannot be seen. `print_plan.md` has the arithmetic.

## Layout

```
index.html            the site — one self-contained file, no network needed
print/                six 3MFs, one per piece, filaments already assigned
src/                  the generator
  paduka_stack.py     the master assembly: cuts the modules, drives the pattern
  paduka_paisley.py   the block library — buta, hashiya, phool, chownk
  paduka_ankle.py     the single-axis joint
  paduka_foot.py      the anatomical foot
  paduka_print.py     inlay, and the 3MF writer
  site/               the web build: patterns, payload, renderer
  print_plan.md       orientation, support, settings, assembly, clearances
```

## Running it

```
pip install trimesh manifold3d shapely numpy pillow matplotlib
cd src
python paduka_stack.py            # cut the modules, report volumes
python paduka_print.py madder     # write the print files
```

The site:

```
cd src/site
python web_pattern.py             # bake the six ornament traditions
python build_payload.py           # geometry + textures + palettes
python build.py                   # one HTML file
```

## Honest status

Works and is measured: the ornament engine, the printed prototype, the ankle's
travel.

Not done: stiffness and energy return have not been measured against
established prosthetic feet; the lime-derived composite the project actually
wants does not exist commercially; clinical validation needs partners.

**This is a printed prototype with correct attachment geometry. It is not a
certified load-bearing prosthesis and is not presented as one.**

Note on `print/`: those six files predate the pass that cut the foot from 82%
of layers carrying ink down to 31%, and from three inks to one. Regenerate
with `paduka_print.py` rather than trusting them for a long print.

## Credit where it is owed

The Jaipur Foot was developed in 1968 by the surgeon P. K. Sethi and the
craftsman Ram Chandra Sharma. Sethi received the Magsaysay Award and the Padma
Shri. Sharma, who made it, was barely credited. A craft prize is the right
place to say that the maker should be named.
