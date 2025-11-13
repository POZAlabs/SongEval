import argparse
import json
from pathlib import Path

import torch
import torchaudio
from hydra.utils import instantiate
from muq import MuQ
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tqdm import tqdm

PATH = Path(__file__).parent.resolve()
DEFAULT_CKPT_PATH = PATH / "ckpt" / "model.safetensors"
DEFAULT_CONFIG_PATH = PATH / "config.yaml"


class SongEval:
    def __init__(self, checkpoint_path: str = None, config_path: str = None):
        self.checkpoint_path = (Path(checkpoint_path) if checkpoint_path else DEFAULT_CKPT_PATH)
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(self.checkpoint_path)
        print(self.config_path)
        self._setup_models()

    @torch.no_grad()
    def _setup_models(self):
        print(f"[SongEval] Loading model from checkpoint: {self.checkpoint_path}")
        print(f"[SongEval] Loading config from: {self.config_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found at: {self.checkpoint_path}"
            )
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")

        train_config = OmegaConf.load(self.config_path)
        self.model = instantiate(train_config.generator).to(self.device).eval()
        state_dict = load_file(self.checkpoint_path, device="cpu")
        self.model.load_state_dict(state_dict, strict=False)

        self.muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        self.muq = self.muq.to(self.device).eval()
        print(f"[SongEval] Models loaded successfully on device: {self.device}")

    @torch.no_grad()
    def evaluate_file(self, audio_path: str) -> dict:
        try:
            wav, sr = torchaudio.load(audio_path)
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                wav = resampler(wav)
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)

            audio = wav.to(self.device)
            output = self.muq(audio, output_hidden_states=True)
            input_features = output["hidden_states"][6]
            scores_g = self.model(input_features).squeeze(0)

            return {
                "Coherence": round(scores_g[0].item(), 4),
                "Musicality": round(scores_g[1].item(), 4),
                "Memorability": round(scores_g[2].item(), 4),
                "Clarity": round(scores_g[3].item(), 4),
                "Naturalness": round(scores_g[4].item(), 4),
            }
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

    def evaluate(self, audio_paths: list[str]) -> dict:
        results = {}
        for file_path in tqdm(audio_paths, desc="Evaluating with SongEval"):
            path_obj = Path(file_path)
            file_id = path_obj.stem
            scores = self.evaluate_file(str(path_obj))
            if scores:
                results[file_id] = scores
        return results

    def run_evaluation(self, input_path: str, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_path = Path(input_path)
        audio_files = []
        if input_path.is_dir():
            audio_files = [str(f) for f in sorted(input_path.glob("*")) if f.is_file()]
        elif input_path.suffix == ".txt":
            with open(input_path, "r", encoding="utf-8") as f:
                audio_files = [line.strip() for line in f if line.strip()]
        elif input_path.is_file():
            audio_files = [str(input_path)]
        else:
            raise FileNotFoundError(
                f"Input path does not exist or is not a valid file/directory: {input_path}"
            )

        results = self.evaluate(audio_files)

        result_path = output_dir / "songeval_results.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"Evaluation complete. Results saved to: {result_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate musical quality of audio files using SongEval.")
    parser.add_argument(
        "-i", "--input_path", type=str, required=True,
        help="Input: Path to a single audio file, a text file listing audio paths, or a directory of audio files.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True,
        help="Output directory for the results JSON file.",
    )
    args = parser.parse_args()

    evaluator = SongEval()
    evaluator.run_evaluation(input_path=args.input_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
