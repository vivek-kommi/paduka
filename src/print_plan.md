# Paduka Stack — print and assembly

Six printed pieces make one 420 mm below-knee limb, with a working single-axis
ankle, standing on an ornamented paduka. Everything below assumes a **Bambu
P1S with the AMS**, a 0.4 mm nozzle, 0.2 mm layers and a 256 × 256 mm plate.

---

## The pieces

| # | File | Piece | Bed footprint | Height | Material |
|---|---|---|---|---|---|
| 1 | `1_paduka.3mf` | sandal | 260 × 96 mm | 44 mm | 281 cc |
| 2 | `2_foot.3mf` | foot | 244 × 81 mm | 76 mm | 454 cc |
| 3 | `3_ankle.3mf` | ankle block | 58 × 59 mm | 60 mm | 86 cc |
| 4 | `4_shank.3mf` | shank | 82 × 87 mm | 184 mm | 543 cc |
| 5 | `5_cuff.3mf` | cuff | 93 × 100 mm | 150 mm | 991 cc |
| 6 | `6_bumpers_and_pins.3mf` | 6 bumpers, 2 pins | 64 × 101 mm | 9 mm | 21 cc |

Files 1–5 open as **one object with several parts**, already assigned to
filaments. Do **not** "split to objects" — the parts are meant to be co-printed
in register, which is the whole point. File 6 is one filament.

---

## The ankle

The stack used to have a rigid ankle. A foot that cannot move at the ankle has
to be walked on by rolling over its toe, which is what makes a cheap
prosthetic foot read as a peg. So the foot now comes apart at the malleoli
into the joint a real single-axis foot has:

| | |
|---|---|
| axis | z = 74 mm — malleolus height, because the malleoli *are* the axis |
| pin | 6 mm, in a 6.3 mm bore, double shear through a 22 mm boss |
| travel | about ±9°, set by the bumpers, not by a hard stop |
| rear bumper | compresses as the heel lands, so the foot comes flat instead of slapping |
| front bumper | compresses as you roll over it and pushes back as you leave — the energy return |

The ankle block is a **column standing in a recess** in the top of the foot,
not a flange on a flat cut. A flat parting plane at this height hands the
ankle block 80 mm of instep as a thin fragile wing; a column in a socket is
both stronger and what an ankle actually looks like. The recess is cut back
8.5 mm all round, which is the column's sweep at full travel.

**The bumpers are the tuning.** Print them in TPU if you have it. In PLA, two
walls and 12% gyroid gives a surprisingly good spring. They are 3 cc each, so
the file gives you six: print a set at 8%, 12% and 20% and walk on the pair
you like.

**The pins lie down on the plate.** A pin is loaded in bending across its
axis; printed standing on end it is a stack of discs held together by layer
adhesion alone. A 6 mm stainless dowel or an M6 bolt is better still.

---

## What decides the ornament

On a multi-material printer, colour is **not** charged per motif. It is
charged **per layer**. Any layer containing two filaments costs one tool
change and one purge, whether it carries a whole paisley or a single dot.

A 420 mm object at 0.2 mm is about 2,100 layers, so an all-over repeat is of
the order of **1.9 kg of purge for a 2.1 kg object**. No flush tuning fixes a
factor of two. So the ornament is **banded**: worked borders — a heavy rule, a
run of butas, a dotted line, a closing rule — with plain single-filament
ground between them. One worked band per module, plus a rule at each seam, a
vamp band across the instep and a welt at the sole.

Each band is laid out by **count**, not pitch. The horizontal axis of the
drawing is an unrolled circumference, so its two ends are the same line — the
seam down the back of the leg. A fixed pitch almost never divides that
circumference evenly, so the last stamp before the seam collides with the
first one after it and a buta comes out sliced in half. Dividing the
circumference into *n* equal cells and centring a stamp in each puts the seam
exactly halfway between two stamps, where nothing is. The foot has a different
circumference from the limb, so it gets its own count. Nothing is ever cut.

Two more decisions follow from the same arithmetic:

- **Two inks on the limb, not three.** The detail block is gone; what it used
  to draw is left as bare ground showing through, which is what a resist print
  *is* — the pattern you see is the dye that did *not* reach the cloth.
- **Nothing is printed in an ink colour where it cannot be seen.** Spigots,
  socket bores, seam flats, collar faces, the dovetail channel, the toe-post
  cleft, and every face of the ankle joint are held at the ground filament. A
  mating face is therefore always one material, so tolerance never drifts with
  colour.

The **paduka keeps all four filaments** and keeps its relief proud. It is
44 mm tall instead of 420, so its polychrome costs almost nothing, and a
sandal you stand on wants ornament that wears in rather than fills up.

---

## Why the ink is 1.5 mm deep, and the lines 1.7 mm wide

The first version had a 0.45 mm inlay and 1.2 mm lines, and it sliced into a
dotted mess: filament 2 printed **5.5 g of model and flushed 71 g**.

On a vertical wall an inlay is only as wide as it is deep, and the slicer lays
perimeters 0.42 mm wide. At 0.45 mm the pattern was thinner than a single
extrusion, so most of it was dropped. The fix is arithmetic, not taste:

| | |
|---|---|
| inlay depth | **1.5 mm** — deeper than the whole wall stack (2 loops = 0.87 mm, 3 = 1.29 mm), so every perimeter inside a motif is ink |
| minimum line | **1.7 mm** — four extrusions across the thinnest thing in the drawing |
| register gap | **1.25 mm** — three extrusions of clear ground between two dyes, or they smear |

Verified on the drawing itself: **0.0% of the ink is thinner than 1.7 mm.**
The same slice now prints 32.5 g of filament 2 instead of 5.5 g.

The pattern is **inlaid, not embossed**, on the limb: the ground carries
recesses and the plugs fill them flush. A plug's outer face *is* the original
surface, so the two can never drift apart, and there is no boolean to fail on
a million-face mesh. Bambu will warn about non-manifold edges on those parts —
that is the ground and the plugs sharing their interface exactly, which is
what flush means. It slices fine.

---

## Orientation and support

- **paduka** — flat, ornament up. 260 mm long on a 256 mm plate, so **rotate
  45°**; the rotated box is about 198 × 203 mm. No support.
- **foot** — sole down, as it sits on the sandal. The dovetail channel, the
  toe-post cleft and the bumper pockets face downward: **support, tree,
  build-plate only**, threshold 30°.
- **ankle block** — fork downward, spigot up. The fork slot and the pin bore
  need **support inside the part** — set support to *everywhere*, not
  build-plate only, for this one piece.
- **shank** and **cuff** — seam face down, spigot up. The flat seam ring is
  the best adhesion surface on the model and every overhang above it is under
  10°. **No support.**
- **bumpers and pins** — flat. Brim on the pins.

---

## Settings that matter on the AMS

| Setting | Value | Why |
|---|---|---|
| Layer height | 0.2 mm | at 0.28 the fine work in the borders drops below one extrusion |
| Wall loops | 2 or 3 | either works; the 1.5 mm inlay covers both |
| Flush multiplier | **1.0** | 1.4 is very conservative; below 0.8 the cream shows navy |
| Flush into support | **on** | the foot and ankle have support anyway |
| Flush into object infill | **off** | it moved 56 g out of the purge but put 185 g of extra plastic into the part — a net loss on this geometry. Test it yourself before trusting either way |
| Prime tower | 45 mm, in a **corner** | the default position collides with the rotated paduka |

Print the pieces **one at a time**. Two multi-material objects on one plate
double the tool changes for no benefit.

---

## Assembly

1. **Bumpers into the foot.** Drop one into each pocket either side of the
   boss. They stand 6 mm proud.
2. **Ankle block onto the foot.** The fork straddles the boss; line up the
   bores and push the pin through. Snug, not tight — the joint should fall
   through its own travel under its own weight.
3. **Foot onto the paduka.** Hold it 20° nose-up, engage the dovetail rail at
   the heel, slide forward until the khadau post comes up through the cleft
   between the great toe and the second. It stops itself. One hand, no tools.
4. **Shank onto the ankle block.** The spigot is an oval, so it only goes on
   one way round. Push down until the shank's seam ring sits on the collar —
   a visible 2 mm shoulder, which is the printed band you can see in the
   pattern. Pin it.
5. **Cuff onto the shank.** Same again.

To change paduka, slide the foot backwards off the rail with the limb still
attached. To break the limb down for a case, pull the three pins.

---

## Clearances

| | |
|---|---|
| ankle boss to fork | 0.35 mm on every face |
| ankle pin | 6.3 mm bore for a 6 mm pin |
| ankle working gap | 6.0 mm at neutral, ±9° of travel |
| ankle recess | 8.5 mm all round — the column's sweep |
| spigot to socket | 0.32 mm |
| dovetail and toe post | 0.35 mm |
| seam pin | 4.4 mm bore for a 4 mm pin |
| register gap between two dyes | 1.25 mm |
| minimum ink width | 1.7 mm |

If your printer runs tight, change the clearance, not the part: `ANK_CLEAR`
in `paduka_ankle.py`, `CLEAR` in `paduka_stack.py`, `MOD.CLEAR` in
`paduka_modular.py`.

---

## If you want it cheaper

`LIMB_BANDS` and `FOOT_BANDS` in `paduka_stack.py` name every band position by
hand. In order of how much they save:

1. **Drop a band.** Deleting either entry in `LIMB_BANDS` takes its layers out
   entirely; deleting `FOOT_BANDS` saves the most.
2. **Print the limb plain and the paduka polychrome.** One filament on the
   limb, the whole wardrobe in the shoes — several paduka in several
   colourways, one foot. This is the project's own argument ("you own one foot
   and several paduka") and it takes the limb's purge to nil. The **Undyed**
   button in the viewer shows exactly this.
