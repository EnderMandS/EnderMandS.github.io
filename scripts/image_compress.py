"""
Convert PNG images under the _posts directory to JPEG to reduce size.
By default:
 - Skips animated PNGs.
 - Skips conversion if JPEG is not at least 'min_ratio' smaller.
 - Removes the original .png after successful conversion.
Optional:
 - Update markdown files to point to new .jpg paths (--update-markdown).

Quality selection guidance:
 - 95: near-lossless, bigger files, diminishing returns above ~92.
 - 85 (default): good balance (typical web baseline).
 - 75: smaller, still acceptable for photos, may show artifacts in flat areas/text.
 - 60 or below: only for thumbnails or when size is critical.
Recommended workflow:
 1. Test a representative PNG: run with --dry-run at several qualities (95,85,75).
 2. Inspect visual difference (zoom to 200%) if concerned.
 3. Keep highest quality that still provides meaningful size drop (e.g. >20% reduction).
Automated idea (not implemented): binary search highest quality producing size <= target % of original.

Progressive JPEG (--progressive):
 - Stores image in multiple scans (low-res to full detail).
 - Enables faster perceived load over slow connections (progressive refinement).
 - Slightly higher CPU to encode, negligible decode difference for modern browsers.
 - File size often a little smaller or similar; occasionally slightly larger.
 Safe to enable for almost all web use.

"""
import argparse
import sys
from pathlib import Path
from io import BytesIO
import re

try:
    from PIL import Image
except ImportError:
    print("Requires Pillow. Install with: pip install Pillow", file=sys.stderr)
    sys.exit(1)

def is_animated(pil_img: Image.Image) -> bool:
    return getattr(pil_img, "is_animated", False) and getattr(pil_img, "n_frames", 1) > 1

def convert_png_to_jpg(png_path: Path, quality: int, progressive: bool, min_ratio: float, force: bool, dry_run: bool):
    orig_size = png_path.stat().st_size
    try:
        with Image.open(png_path) as im:
            if is_animated(im):
                return False, "skip(animated)"
            # Convert / flatten
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
                im_out = bg
            else:
                im_out = im.convert("RGB")
            # Encode to memory first
            buf = BytesIO()
            im_out.save(
                buf,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=progressive,
            )
            jpg_bytes = buf.getvalue()
    except Exception as e:
        return False, f"error({e})"

    jpg_size = len(jpg_bytes)
    # Ensure size reduction unless force
    if not force:
        if jpg_size >= orig_size * min_ratio:
            return False, f"skip(no_gain {orig_size} -> {jpg_size})"

    jpg_path = png_path.with_suffix(".jpg")
    if dry_run:
        return True, f"would_convert({orig_size}->{jpg_size})"

    # Write JPEG
    with open(jpg_path, "wb") as f:
        f.write(jpg_bytes)

    # Remove original PNG
    png_path.unlink(missing_ok=True)
    return True, f"ok({orig_size}->{jpg_size})"

def update_markdown_links(md_paths, converted_png_abs_set, dry_run: bool, verbose: int, legacy_map=None):
    r"""
    Smart replacement:
      For every token matching [\w./-]+\.png in a markdown file, resolve it
      relative to the markdown file directory. If that absolute path is in the
      converted set, change only the extension to .jpg.
    If legacy_map (dict) is provided, apply the old broad string replacement after.
    """
    total_files_changed = 0
    total_refs_changed = 0
    pattern = re.compile(r'(?P<path>[\w./-]+\.png)', re.IGNORECASE)
    referenced_converted = set()  # newly tracked

    for md in md_paths:
        try:
            text = md.read_text(encoding="utf-8")
        except Exception:
            continue
        original = text
        def repl(m):
            nonlocal total_refs_changed
            ref = m.group('path')
            candidate = (md.parent / ref).resolve()
            if candidate in converted_png_abs_set:
                total_refs_changed += 1
                referenced_converted.add(candidate)
                return ref[:-4] + '.jpg'
            return ref
        text = pattern.sub(repl, text)
        # legacy broad replacement (optional)
        if legacy_map:
            for old_rel, new_rel in legacy_map.items():
                text = text.replace(old_rel, new_rel)
        if text != original:
            total_files_changed += 1
            if not dry_run:
                md.write_text(text, encoding="utf-8")
            if verbose:
                print(f"MD UPDATED {md.relative_to(md.parents[2]) if len(md.parents) > 2 else md}: refs={total_refs_changed}")
    return total_files_changed, total_refs_changed, referenced_converted

def gather_markdown_files(root: Path):
    return [p for p in root.rglob("*.md")]

def main():
    parser = argparse.ArgumentParser(description="Convert PNG images under _posts to JPEG.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing _posts/")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (0-100, default 85). 75-85 is a common web range.")
    parser.add_argument("--min-ratio", type=float, default=0.98,
                        help="Require JPEG size < orig_size * min_ratio (default 0.98). Use 1.0 to accept any reduction.")
    parser.add_argument("--progressive", action="store_true", help="Write progressive (interlaced) JPEG for incremental rendering.")
    parser.add_argument("--force", action="store_true", help="Force conversion even if not smaller")
    parser.add_argument("--update-markdown", action="store_true", help="Rewrite .md references from .png to .jpg")
    parser.add_argument("--legacy-replace", action="store_true", help="Use legacy global string replacement method (less precise).")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without writing files")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    posts_dir = args.root / "_posts"
    if not posts_dir.is_dir():
        print(f"_posts directory not found at: {posts_dir}", file=sys.stderr)
        sys.exit(1)

    png_files = [p for p in posts_dir.rglob("*.png")]
    if not png_files:
        print("No PNG files found.")
        return

    converted = 0
    skipped = 0
    converted_png_abs = set()
    legacy_map = {}  # only populated if --legacy-replace
    for png in png_files:
        ok, msg = convert_png_to_jpg(
            png,
            quality=args.quality,
            progressive=args.progressive,
            min_ratio=args.min_ratio,
            force=args.force,
            dry_run=args.dry_run,
        )
        rel = png.relative_to(args.root)
        if args.verbose:
            print(f"{'CONVERT' if ok else 'SKIP'} {rel} -> {msg}")
        if ok:
            converted += 1
            if args.update_markdown:
                if args.legacy_replace:
                    old_rel = str(rel).replace("\\", "/")
                    new_rel = old_rel[:-4] + ".jpg"
                    legacy_map[old_rel] = new_rel
                converted_png_abs.add(png.resolve())
        else:
            skipped += 1

    md_files_changed = 0
    refs_changed = 0
    if args.update_markdown and converted_png_abs:
        md_files = gather_markdown_files(posts_dir)
        md_files_changed, refs_changed, referenced_converted = update_markdown_links(
            md_files,
            converted_png_abs,
            args.dry_run,
            args.verbose,
            legacy_map if args.legacy_replace else None
        )
    else:
        referenced_converted = set()

    unused_converted = sorted(converted_png_abs - referenced_converted)
    print(f"Done. Converted: {converted}, Skipped: {skipped}, Markdown files updated: {md_files_changed}, Links changed: {refs_changed}, Dry-run: {args.dry_run}")
    if unused_converted:
        print(f"Unused converted images (no markdown reference found): {len(unused_converted)}")
        for p in unused_converted:
            try:
                print(p.relative_to(args.root))
            except ValueError:
                print(p)
    else:
        print("No unused converted images.")

if __name__ == "__main__":
    main()
