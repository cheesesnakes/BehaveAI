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

from tqdm.asyncio import tqdm

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
        default="samples",
        help="Directory to store intermediate clips (default: samples).",
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
    clip_info = {}
    if CSV_PATH is None or not os.path.exists(CSV_PATH):
        print("CSV file not found, defaulting to all videos in source directory.")
        for file in os.listdir(SOURCE_DIR):
            if file.endswith(SOURCE_EXT):
                file_name = os.path.splitext(file)[0]
                start_s = round(uniform(0, 9 * 60))
                clip_info[file_name] = {"file_name": file_name, "sample_s": start_s}
        return clip_info

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if "deploymen_id" in row.keys():
                sample_id = (
                    row["deployment_id"].strip()
                    + row["plot_id"].strip()
                    + row["sample_id"].strip()
                )
            else:
                sample_id = row["sample_id"]
            if "clip_time" in row.keys():
                sample_s = float(row["clip_time"].split(":")[0]) * 60 + float(
                    row["clip_time"].split(":")[1]
                )

            else:
                sample_s = float(row["start_s"])

            clip_info[sample_id] = {
                "file_name": row["file_name"].strip(),
                "sample_s": sample_s,
            }

    return clip_info


# cut clips
def cut_clip():
    global CSV_PATH, SOURCE_DIR, SOURCE_EXT, CLIP_DURATION, WORK_DIR

    clip_paths = []
    failed = []
    reader = clips()

    for i, row in reader.items():
        print(f"Processing row {i}: {row}")
        file_name = row["file_name"].strip()
        start_s = float(row["sample_s"])
        if file_name[-4:] == SOURCE_EXT:
            file_name = file_name[:-4]
        source = f"{SOURCE_DIR}/{file_name}{SOURCE_EXT}"

        if not os.path.exists(source):
            print(f"  ! missing source: {source}, skipping")
            failed.append((i, source, "missing source file"))
            continue

        if not os.path.exists(WORK_DIR):
            os.makedirs(WORK_DIR)

        if file_name.count("/") > 0:
            file_name = file_name.split("/")[-1]
            out_clip = f"{WORK_DIR}/{i}.mp4"
        else:
            out_clip = f"{WORK_DIR}/{file_name}_sampled.mp4"

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_s),
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
            "-fps_mode",
            "cfr",
            "-r",
            "30",
            "-progress",
            "pipe:1",
            "-nostats",
            str(out_clip),
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,  # capture stderr so we can report it on failure
                text=True,
                bufsize=1,
            )

            pbar = tqdm(total=round(CLIP_DURATION, 2), unit="s", desc=f"Encoding {i}")
            current_time = 0.0

            for line in process.stdout:
                line = line.strip()
                if line.startswith("out_time_ms="):
                    ms = line.split("=")[1]
                    if ms.strip() == "N/A":
                        continue
                    new_time = round(int(ms) / 1_000_000, 2)
                    pbar.update(new_time - current_time)
                    current_time = new_time
                elif line == "progress=end":
                    pbar.update(CLIP_DURATION - current_time)
                    break

            pbar.close()
            process.wait()

            # check ffmpeg exit code
            if process.returncode != 0:
                stderr_output = process.stderr.read()
                failed.append(
                    (
                        i,
                        source,
                        f"ffmpeg exited with code {process.returncode}: {stderr_output[:200]}",
                    )
                )
                continue

            # check the output file actually has content
            if not os.path.exists(out_clip) or os.path.getsize(out_clip) == 0:
                failed.append((i, source, "output file is empty or missing"))
                continue

            clip_paths.append(out_clip)

        except Exception as e:
            failed.append((i, source, str(e)))
            continue

    # report failures at the end
    if failed:
        print(f"\n{'=' * 40}")
        print(f"  {len(failed)} clip(s) failed:")
        for row_i, src, reason in failed:
            print(f"  row {row_i} | {src}")
            print(f"    reason: {reason}")
        print(f"{'=' * 40}\n")
    else:
        print("\nAll clips processed successfully.")

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
