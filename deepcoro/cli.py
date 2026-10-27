"""Command line entry point; heavy inference dependencies load after parsing."""
import argparse
import logging


def main(args=None):
    parser = argparse.ArgumentParser(description="DeepCoro coronary stenosis inference")
    parser.add_argument("--input-file-path", "--input_file_path", default="/dcm_input/input_file.csv")
    parser.add_argument("--save-dir", "--save_dir", default="/results/")
    parser.add_argument("--models-dir", "--models_dir", default=None,
                        help="Local model directory; otherwise download from Hugging Face")
    parser.add_argument("--repo-id", default="heartwise/DeepCoro")
    parser.add_argument("--revision", help="Hugging Face commit or branch (default: pinned official snapshot)")
    parser.add_argument("--cache-dir", help="Hugging Face cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="Use cached weights without network access")
    parser.add_argument("--device", help="PyTorch device, for example cpu or cuda:0")
    parser.add_argument("--no-videos", action="store_true", help="Write predictions CSV without annotated videos")
    options = parser.parse_args(args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from .pipeline import deepcoro
    deepcoro(options.input_file_path, options.save_dir, options.models_dir,
             repo_id=options.repo_id, revision=options.revision, cache_dir=options.cache_dir,
             local_files_only=options.local_files_only, device=options.device,
             save_videos=not options.no_videos)
    return 0
