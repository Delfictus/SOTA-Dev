#!/usr/bin/env python3
"""
Give each figure panel in the PRISM4D DOCX its own dedicated full page.

For each inline image:
  1. Set page_break_before=True on the image paragraph (starts a fresh page).
  2. Center-align the image paragraph.
  3. Resize the image to fill the usable page area (maintain aspect ratio,
     fit within usable_width × (usable_height - caption_reserve)).
  4. Insert a page-break paragraph immediately after the caption so the
     following text starts on a new page.

Existing empty page-break paragraphs immediately after captions are removed
to avoid creating blank pages.
"""

from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

INPUT  = Path("/mnt/storage/prism-outputs/docx_export_20260514T080725Z/"
              "PRISM4D_final_structural_integrated_v3.docx")
OUTPUT = Path("/mnt/storage/prism-outputs/docx_export_20260514T080725Z/"
              "PRISM4D_final_structural_integrated_v4_structural_fullpage.docx")

CAPTION_RESERVE_IN = 0.65   # inches reserved for caption + vertical spacing
EMU_PER_INCH       = 914400

# Structural PyMOL renders are portrait (aspect < 1.0).
# Landscape charts/graphs (aspect > 1.0) are left completely untouched.
STRUCTURAL_ASPECT_THRESHOLD = 1.0


def make_page_break_paragraph() -> "lxml.etree._Element":
    """Return a bare <w:p> containing a page-break run."""
    p   = OxmlElement("w:p")
    r   = OxmlElement("w:r")
    br  = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    r.append(br)
    p.append(r)
    return p


def is_empty_page_break_para(para) -> bool:
    """True if paragraph has no visible text and contains a page-break run."""
    if para.text.strip():
        return False
    xml = para._element.xml
    return 'w:type="page"' in xml or "w:type='page'" in xml


def find_caption_para(paras, img_para_idx: int):
    """
    Return the caption paragraph that follows image para at img_para_idx.
    Scans up to 4 paragraphs forward; returns None if not found.
    Caption = style 'Caption'  OR  text starts with 'Figure'.
    """
    for offset in range(1, 5):
        j = img_para_idx + offset
        if j >= len(paras):
            break
        p = paras[j]
        if p.style.name == "Caption":
            return j, p
        txt = p.text.strip()
        if txt.lower().startswith("figure") and len(txt) > 6:
            return j, p
        # Stop searching if we hit non-empty non-image content that isn't a caption
        blip = p._element.findall(".//" + qn("a:blip"))
        if txt and not blip:
            break
    return None, None


def main():
    doc = Document(INPUT)
    sec = doc.sections[0]

    usable_w = sec.page_width  - sec.left_margin  - sec.right_margin
    usable_h = (sec.page_height - sec.top_margin   - sec.bottom_margin
                - int(CAPTION_RESERVE_IN * EMU_PER_INCH))

    paras = doc.paragraphs
    # snapshot element→index map BEFORE we mutate the XML tree
    elem_to_idx = {p._element: i for i, p in enumerate(paras)}

    # Collect (shape, img_para_idx) pairs
    shape_info = []
    for shape in doc.inline_shapes:
        # wp:inline → w:drawing → w:r → w:p
        try:
            r_elem = shape._inline.getparent().getparent()
            p_elem = r_elem.getparent()
            pidx   = elem_to_idx.get(p_elem)
            if pidx is not None:
                shape_info.append((shape, pidx))
        except Exception as e:
            print(f"  WARNING: could not locate shape in paragraph tree: {e}")

    print(f"Found {len(shape_info)} figures to process.")
    print(f"Usable area: {usable_w/EMU_PER_INCH:.3f}\" x {usable_h/EMU_PER_INCH:.3f}\"")

    # Track which paragraphs we already processed (multiple shapes can share a para)
    processed_img_paras = set()

    for shape, pidx in shape_info:
        img_para = paras[pidx]

        # ── Skip landscape charts/graphs — only touch structural renders ──────
        aspect_check = shape.width / shape.height if shape.height else 999
        if aspect_check >= STRUCTURAL_ASPECT_THRESHOLD:
            print(f"  Fig in para[{pidx:3d}]: landscape (aspect={aspect_check:.2f}) — skip")
            continue

        # ── 1. Resize to fill page (maintain aspect ratio) ───────────────────
        if shape.height and shape.height > 0:
            aspect     = shape.width / shape.height
            page_ratio = usable_w / usable_h
            if aspect >= page_ratio:        # landscape / wider: width-constrained
                new_w = usable_w
                new_h = int(usable_w / aspect)
            else:                           # portrait / taller: height-constrained
                new_h = usable_h
                new_w = int(usable_h * aspect)
            shape.width  = new_w
            shape.height = new_h
            print(f"  Fig in para[{pidx:3d}]: resized to "
                  f"{new_w/EMU_PER_INCH:.3f}\" x {new_h/EMU_PER_INCH:.3f}\"")
        else:
            print(f"  Fig in para[{pidx:3d}]: zero height — skip resize")

        if pidx in processed_img_paras:
            continue
        processed_img_paras.add(pidx)

        # ── 2. Page-break-before + centre the image paragraph ────────────────
        img_para.paragraph_format.page_break_before = True
        img_para.paragraph_format.alignment         = WD_ALIGN_PARAGRAPH.CENTER

        # ── 3. Find caption ───────────────────────────────────────────────────
        cap_idx, cap_para = find_caption_para(paras, pidx)

        # ── 4. Centre-align the caption ───────────────────────────────────────
        if cap_para:
            cap_para.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER

        # ── 5. Remove any existing empty page-break paragraph right after ─────
        #      caption (or image if no caption) to avoid blank pages.
        anchor_idx = cap_idx if cap_para else pidx
        anchor_para = cap_para if cap_para else img_para
        if anchor_idx is not None:
            next_idx = anchor_idx + 1
            if next_idx < len(paras) and is_empty_page_break_para(paras[next_idx]):
                paras[next_idx]._element.getparent().remove(paras[next_idx]._element)
                print(f"    removed existing blank page-break para[{next_idx}]")

        # ── 6. Insert page break after caption (or after image) ───────────────
        pb_para = make_page_break_paragraph()
        anchor_para._element.addnext(pb_para)
        print(f"    inserted page-break after para[{anchor_idx}] "
              f"({'caption' if cap_para else 'image'})")

    doc.save(OUTPUT)
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()
