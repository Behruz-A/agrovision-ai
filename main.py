"""AgroVision AI command-line application."""
import argparse
from pathlib import Path

from src.predict import WorkbenchPredictor, webcam
from src.utils import default_checkpoint, write_json


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="AgroVision AI: identify vegetables or analyze supported crop leaves.")
    parser.add_argument("--mode", default="gui", choices=["gui", "image", "webcam"], help="Default: gui (image picker window)")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--task", choices=["auto", "vegetable", "leaf"], default="auto")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="Optional image prediction JSON path")
    args = parser.parse_args(argv)
    if args.mode == "image" and args.image is None:
        parser.error("--image is required with --mode image")
    try:
        if args.mode == "gui":
            from src.gui import launch_gui
            launch_gui(args.checkpoint, args.device, args.image, "leaf" if args.task == "leaf" else "vegetable")
            return
        predictor = WorkbenchPredictor(args.checkpoint, args.device)
        if args.mode == "webcam":
            webcam(predictor, args.camera, args.interval, None if args.task == "auto" else args.task)
        else:
            result = predictor.predict_file(args.image, None if args.task == "auto" else args.task)
            print("AgroVision AI\n" + "-" * 30)
            if result["smoke_test_model"]:
                print("WARNING: smoke-test model; predictions are not validated for practical use.")
            print(f"Prediction: {result['prediction']}\nModel estimate: {result['confidence']:.1%}")
            if result.get("warning"):
                print(result["warning"])
            for name, probability in result["probabilities"].items():
                print(f"  {name}: {probability:.2%}")
            if args.output:
                write_json(args.output, result)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"AgroVision AI: {exc}\n")


if __name__ == "__main__":
    main()
