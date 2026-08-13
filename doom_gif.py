"""Assemble doom/frames/*.png into an animated GIF.

    .venv/bin/python doom_gif.py [--out doom/doom.gif] [--every 1] [--ms 100]
"""
import argparse
import pathlib

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="doom/frames")
    ap.add_argument("--out", default="doom/doom.gif")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--ms", type=int, default=40,
                    help="per-frame delay; 1 frame = 1 game tic (1/35 s)")
    ap.add_argument("--start", type=int, default=0,
                    help="first frame index (skip the title screen)")
    args = ap.parse_args()

    paths = sorted(pathlib.Path(args.frames_dir).glob("frame_*.png"))
    paths = paths[args.start::args.every]
    if not paths:
        print("no frames yet")
        return
    frames = [Image.open(p).convert("P") for p in paths]
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=args.ms, loop=0, optimize=True)
    print(f"{len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
