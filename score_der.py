#!/usr/bin/env python3
"""Score diarization error rate between two RTTM directories.

``--base-path`` is the reference folder and ``--output-path`` is the hypothesis
folder. Both look like ``.../metadatas``: one NIST RTTM per recording, paired
by filename.

::

    SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>

Times are seconds. The speaker name is field 8. Hypothesis labels do not need
to match the reference; pyannote maps them with the Hungarian algorithm.

Before scoring, both sides go through the same merge-filter used by
``diarization/diarize.py`` (``min_off=1e9``, ``min_on=0.5``). The metric is
``pyannote.metrics.diarization.DiarizationErrorRate`` with collar 0 and
overlap regions kept.

Example
-------
::

    .venv/bin/python score_der.py \\
      --base-path /home/duyn/ActableDuy/datasets/PhoAudioBook-complete-diar/metadatas \\
      --output-path /path/to/hypothesis/metadatas
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# No audio duration is available from an RTTM directory, so pyannote builds
# the evaluation map from the reference and hypothesis extents. Same call as
# diarization/diarize.py, which also omits an explicit UEM.
warnings.filterwarnings(
    "ignore",
    message=r"'uem' was approximated by the union of 'reference' and 'hypothesis' extents\.",
    category=UserWarning,
)

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from tqdm import tqdm


def read_rttm(path: Path) -> list[tuple[float, float, str]]:
    """NIST SPEAKER lines to ``(start, end, speaker)`` in seconds.

    Same column layout as ``diarization/utilities.py``: type, file id, channel,
    start, duration, orthography, speaker type, speaker name, confidence, slate.
    """
    segments: list[tuple[float, float, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] != "SPEAKER":
                continue
            if len(parts) < 10:
                raise ValueError(f"{path}:{line_no}: expected 10 RTTM fields, got {len(parts)}")
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            if end < start:
                raise ValueError(f"{path}:{line_no}: negative duration {duration}")
            segments.append((start, end, parts[7]))
    return segments


def merge_n_filter_post_segments(
    segments: list[tuple[float, float, str]],
    min_off: float,
    min_on: float,
) -> list[tuple[float, float, str]]:
    """Merge same-speaker gaps shorter than ``min_off``, drop turns shorter than ``min_on``.

    Matches ``diarization/utilities.py``. Consecutive same-speaker turns are
    merged only when nothing else sits between them in time order. Returns an
    empty list when every turn is filtered out.
    """
    if not segments:
        return []

    segments = sorted(segments, key=lambda item: (item[0], item[1], str(item[2])))
    merged: list[tuple[float, float, str]] = []
    cur_start, cur_end, cur_spk = segments[0]
    for start, end, spk in segments[1:]:
        if spk == cur_spk and (start - cur_end) < min_off:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end, cur_spk))
            cur_start, cur_end, cur_spk = start, end, spk
    merged.append((cur_start, cur_end, cur_spk))

    filtered = [(start, end, spk) for start, end, spk in merged if (end - start) >= min_on]
    if not filtered:
        return []

    merged = []
    cur_start, cur_end, cur_spk = filtered[0]
    for start, end, spk in filtered[1:]:
        if spk == cur_spk and (start - cur_end) < min_off:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end, cur_spk))
            cur_start, cur_end, cur_spk = start, end, spk
    merged.append((cur_start, cur_end, cur_spk))
    return merged


def segments_to_annotation(
    segments: list[tuple[float, float, str]],
    uri: str,
) -> Annotation:
    annotation = Annotation(uri=uri)
    for start, end, speaker in segments:
        if end > start:
            annotation[Segment(float(start), float(end))] = str(speaker)
    return annotation


def list_rttm(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")
    files: dict[str, Path] = {}
    for path in sorted(directory.glob("*.rttm")):
        if not path.is_file():
            continue
        if path.name in files:
            raise SystemExit(f"Duplicate RTTM name in {directory}: {path.name}")
        files[path.name] = path
    if not files:
        raise SystemExit(f"No .rttm files in {directory}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        required=True,
        help="Reference RTTM directory (e.g. .../metadatas)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Hypothesis RTTM directory, matched to the reference by filename",
    )
    parser.add_argument(
        "--collar",
        type=float,
        default=0.0,
        help="pyannote DER collar in seconds (default 0.0)",
    )
    parser.add_argument(
        "--skip-overlap",
        action="store_true",
        help="Skip overlapping speech regions in DER (default: score them)",
    )
    parser.add_argument(
        "--min-off",
        type=float,
        default=1e9,
        help="Merge same-speaker gaps shorter than this many seconds (default 1e9)",
    )
    parser.add_argument(
        "--min-on",
        type=float,
        default=0.5,
        help="Drop turns shorter than this many seconds (default 0.5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N paired files, in filename order",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Print one DER line per file",
    )
    return parser.parse_args()


def _rate(part: float, total: float) -> float:
    if total <= 0.0:
        return 0.0
    return part / total


def main() -> None:
    args = parse_args()
    reference_files = list_rttm(args.base_path)
    hypothesis_files = list_rttm(args.output_path)

    paired = [name for name in reference_files if name in hypothesis_files]
    missing_hyp = [name for name in reference_files if name not in hypothesis_files]
    missing_ref = [name for name in hypothesis_files if name not in reference_files]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit(f"--limit must be >= 1, got {args.limit}")
        paired = paired[: args.limit]

    print(f"reference : {args.base_path} ({len(reference_files)} rttm)")
    print(f"hypothesis: {args.output_path} ({len(hypothesis_files)} rttm)")
    print(f"paired    : {len(paired)}")
    print(f"post      : min_off={args.min_off} min_on={args.min_on}")
    print(f"DER       : collar={args.collar} skip_overlap={args.skip_overlap}")

    metric = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
    ders: list[float] = []
    errors: list[tuple[str, str]] = []

    for name in tqdm(paired, unit="file"):
        uri = Path(name).stem
        try:
            ref_segments = merge_n_filter_post_segments(
                read_rttm(reference_files[name]),
                min_off=args.min_off,
                min_on=args.min_on,
            )
            hyp_segments = merge_n_filter_post_segments(
                read_rttm(hypothesis_files[name]),
                min_off=args.min_off,
                min_on=args.min_on,
            )
            components = metric(
                segments_to_annotation(ref_segments, uri),
                segments_to_annotation(hyp_segments, uri),
                detailed=True,
                uri=uri,
            )
            der = float(components["diarization error rate"])
        except Exception as exc:
            errors.append((name, str(exc)))
            tqdm.write(f"ERROR {name}: {exc}")
            continue

        ders.append(der)
        if args.per_file:
            tqdm.write(
                f"{uri}  DER {der:.4f}  "
                f"ref {len({spk for *_, spk in ref_segments})}spk/{len(ref_segments)}seg  "
                f"hyp {len({spk for *_, spk in hyp_segments})}spk/{len(hyp_segments)}seg"
            )

    total = float(metric["total"]) if metric.results_ else 0.0
    false_alarm = float(metric["false alarm"]) if metric.results_ else 0.0
    missed = float(metric["missed detection"]) if metric.results_ else 0.0
    confusion = float(metric["confusion"]) if metric.results_ else 0.0
    global_der = float(abs(metric)) if metric.results_ else None
    mean_der = sum(ders) / len(ders) if ders else None

    print("\n========== SUMMARY ==========")
    print(f"scored       : {len(ders)}")
    print(f"errors       : {len(errors)}")
    print(f"missing hyp  : {len(missing_hyp)}")
    print(f"extra hyp    : {len(missing_ref)}")
    if mean_der is not None:
        print(f"Mean DER     : {mean_der:.4f}")
    if global_der is not None:
        print(f"Global DER   : {global_der:.4f}")
        print(f"  false alarm: {_rate(false_alarm, total):.4f}")
        print(f"  missed det : {_rate(missed, total):.4f}")
        print(f"  confusion  : {_rate(confusion, total):.4f}")

    def _preview(title: str, names: list[str]) -> None:
        if not names:
            return
        shown = ", ".join(names[:10])
        suffix = "" if len(names) <= 10 else f" ... (+{len(names) - 10})"
        print(f"{title}: {shown}{suffix}")

    _preview("missing hyp", missing_hyp)
    _preview("extra hyp", missing_ref)

    if not ders:
        sys.exit(1)


if __name__ == "__main__":
    main()
