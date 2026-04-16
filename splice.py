#!.venv/bin/env python3
"""
Splice video clips based on a CSV of start times.
For each row: cut a 2-minute clip starting at sample_s from the source video,
then concatenate everything into one final video.
"""

import argparse
import csv
import os
import subprocess
from random import uniform

# ----------------


def config_parse():
    parser = argparse.ArgumentParser(
        description="Splice video clips based on a CSV of start times."
    )
    parser.add_argument("--csv", default=None, help="Path to the input CSV file.")
    parser.add_argument(
        "--source-dir", required=True, help="Directory containing source videos."
    )
    parser.add_argument(
        "--source-ext",
        default=".mkv",
        help="Extension of source video files (default: .mkv).",
    )
    parser.add_argument(
        "--clip-duration",
        type=int,
        default=120,
        help="Duration of each clip in seconds (default: 120).",
    )
    parser.add_argument(
        "--output-file",
        default="final_output.mp4",
        help="Filename for the final concatenated video (default: final_output.mp4).",
    )
    parser.add_argument(
        "--work-dir",
        default="clips_workdir",
        help="Directory to store intermediate clips (default: clips_workdir).",
    )

    parser.add_argument(
        "--concatenate",
        action="store_true",
        default=False,
        help="Whether to concatenate clips into a final video.",
    )

    return parser.parse_args()


# read csv or create list oof clips from SOURCE_DIR


def clips():
    global CSV_PATH, SOURCE_DIR, SOURCE_EXT

    if CSV_PATH is None or not os.path.exists(CSV_PATH):
        clip_info = {}
        print("CSV file not found, defaulting to all videos in source directory.")
        for file in os.listdir(SOURCE_DIR):
            if file.endswith(SOURCE_EXT):
                file_name = os.path.splitext(file)[0]
                start_s = round(uniform(0, 9 * 60))
                clip_info[file_name] = {"file_name": file_name, "sample_s": start_s}
        return clip_info

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        return reader


# cut clips
def cut_clip():
    global CSV_PATH, SOURCE_DIR, SOURCE_EXT, CLIP_DURATION, WORK_DIR

    clip_paths = []
    reader = clips()
    print(reader)
    # Assumes columns: deployment, sample_s, predator_present, completed
    for (
        i,
        row,
    ) in reader.items() if isinstance(reader, dict) else enumerate(reader):
        print(f"Processing row {i}: {row}")
        # Extract info from the row
        file_name = row["file_name"].strip()
        start_s = float(row["sample_s"])
        source = f"{SOURCE_DIR}/{file_name}{SOURCE_EXT}"

        # Check source exists, skip if missing
        if not os.path.exists(source):
            print(f"  ! missing source: {source}, skipping")
            continue

        # Define output clip path
        if not os.path.exists(WORK_DIR):
            os.makedirs(WORK_DIR)

        out_clip = f"{WORK_DIR}/{file_name}_sampled.mp4"

        # Re-encode so all clips have matching codecs/timebases for clean concat.
        # Slower than stream-copy but far more reliable across mixed sources.
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_s),  # seek before -i = fast seek
            "-i",
            str(source),
            "-t",
            str(CLIP_DURATION),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-vsync",
            "cfr",
            "-r",
            "30",  # force constant frame rate
            str(out_clip),
        ]

        # Debug print for progress tracking
        print(f"[{i}] {file_name} cut @ {start_s}s -> {out_clip}")

        # Run the command, suppressing output for cleaner logs
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        # Add the successfully created clip to the list

        clip_paths.append(out_clip)

    return clip_paths


# concatenate all the clips into one final video
def concatenate_clips(clip_paths, output_file):
    global WORK_DIR, OUTPUT_FILE

    # Build the concat list file
    concat_list = WORK_DIR / "concat_list.txt"
    #
    with open(concat_list, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p.resolve()}'\n")

    # Concatenate (stream copy is fine since all clips were re-encoded identically)
    print(f"\nConcatenating {len(clip_paths)} clips -> {OUTPUT_FILE}")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            OUTPUT_FILE,
        ],
        check=True,
    )
    return 0


def main():
    global \
        CSV_PATH, \
        SOURCE_DIR, \
        SOURCE_EXT, \
        CLIP_DURATION, \
        OUTPUT_FILE, \
        WORK_DIR, \
        CONCATENATE

    args = config_parse()
    CSV_PATH = args.csv
    SOURCE_DIR = os.path.abspath(args.source_dir)
    SOURCE_EXT = args.source_ext
    CLIP_DURATION = args.clip_duration
    OUTPUT_FILE = args.output_file
    WORK_DIR = os.path.abspath(args.work_dir)
    CONCATENATE = args.concatenate

    clips = cut_clip()

    if not clips:
        print("No clips were created. Check your CSV and filters.")
        return 1

    if CONCATENATE:
        concatenate_clips(clips, OUTPUT_FILE)

    print("Done!")

    return 0


if __name__ == "__main__":
    main()
