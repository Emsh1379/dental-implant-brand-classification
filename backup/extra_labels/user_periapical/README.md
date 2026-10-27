# User-supplied Bego radiographs (bego.zip, 2026-10-27), auto-labelled

`bego.zip` had 19 files: 2 exact duplicates, and a mix of periapical radiographs, panoramics, phone photos of panoramic films and social-media collages.
Used: the **10 unique periapical radiographs** (matching the dataset's image type), 16 implants. Not used: the two panoramics, two phone photos of films
(blue-tinted), three social-media composites with text overlays (one has two tiny periapical thumbnails), and one journal figure (`1-s2.0-S2214541926000349-gr9_lrg.jpg`, copyrighted).

| file | source | implants |
|---|---|---|
| ubego-00 | Screenshot 2026-10-26 013414.png | 2 |
| ubego-01 | photo_…349_y.jpg | 2 |
| ubego-02 | photo_…350_y.jpg (BEGO logo) | 2 (rough boxes placed by hand, snapped to the metal – the detectors found nothing) |
| ubego-03 | photo_…351_y.jpg | 3 |
| ubego-04 | photo_…352_y.jpg | 2 |
| ubego-05 … 09 | photo_…354 … 358_y.jpg | 1 each (05–09 look like the same patient; 05 and 06 may be the same site at two times) |

Images are grey, stretched to 640 × 640 like the Roboflow export. Boxes: 5-model YOLO11n consensus (≥ 3 of 5), overlapping boxes reduced to one, snapped to
the implant metal; checked visually (`contact_sheet.jpg`, `preview/`). Label = Bego for all (user's statement; several images carry the BEGO logo).

**The detectors voted ITI for most of these implants** (majority ITI on 12 of the 14 detected implants, all 5 models ITI on 7; only ubego-00's two implants got a Bego majority). These are probably a different Bego implant line
from the Bego implants in the dataset, which is the Bego/ITI confusion the classifier already has.

Provenance: images collected from the internet / messaging apps. Check the right to use them before publishing; describe them as auto-labelled, user-collected data.
