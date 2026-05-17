# Hand Gesture Detector

This project collects hand landmark data, trains a small neural network, and uses the trained model to detect gestures from a live camera feed.

## Project structure

- `data/train/` - training CSV files
- `data/test/` - test CSV files
- `models/` - saved model weights and config
- `register_cords/` - script for collecting labeled hand landmark samples
- `learn_model/` - script for training the model
- `detector/` - live camera detector

## Requirements

The project uses Python and the packages listed in each module's `requirements.txt` file. The detector also needs a working camera and a display environment.

## Collect data

Use the sample capture script to record new gestures.

Before running it, edit `register_cords/register_cords.py` and set `GESTURE_LABEL` to the label you want to save.

```bash
cd register_cords
bash run.sh
```

Press `S` to save the current hand landmarks to a CSV file. Press `Esc` to exit.

## Train the model

Training reads the CSV files from `data/train/` and `data/test/`, then saves the model to `models/`.

```bash
cd learn_model
bash run.sh
```

To force a rebuild of the Docker image, run:

```bash
bash run.sh --build
```

## Run live detection

After training, start the detector from the project root.

```bash
cd detector
bash run.sh
```

The detector opens the camera, draws the hand landmarks, and prints the predicted gesture when the confidence checks pass.

## Notes

- Training and detection expect each sample row to contain 63 numeric values followed by the label.
- The saved model files are `models/gesture_mlp.pth` and `models/gesture_mlp.config.json`.
- If you use the Docker scripts, make sure Docker has access to the camera and, for the detector, the X display.

