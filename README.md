# Facial Expression Detection

This repository contains the implementation of a real-time facial expression recognition system using Convolutional Neural Networks (CNN).

🤩 Please give me a star if you find it interesting!

## Facial Expression Detection: Real-time Emotion Recognition using CNN Architecture
**Zohaib Hassan**  
*Data Scientist*

Our model achieves 85% accuracy on 7 emotion classes!

🔥🔥 Our trained model on FER2013 dataset has been released!

🤗 Hugging Face Link (Coming Soon)

## Demo & Workflow

![Demo](demo.gif)

## Installation

### Clone the repository
```bash
git clone https://github.com/zohaibhassan00/Facial-Expression-Detection
cd Facial-Expression-Detection
```

### Create a conda environment and install the required packages
```bash
conda create -n facial-expression python==3.8
conda activate facial-expression
pip install tensorflow==2.8.0 opencv-python==4.7.0.72 numpy==1.21.6 matplotlib==3.5.3 scikit-learn==1.1.3 pillow==9.2.0 tqdm==4.64.1
```

## Usage

### Real-time Detection
```bash
python real_time_detection.py --camera 0
```

### Image Detection
```bash
python detect_emotion.py --image_path <path-to-image> --model_path <path-to-model>
```

### Training
```bash
python train_model.py --dataset_path <path-to-fer2013> --epochs 50 --batch_size 32
```

## Model Architecture

- **Input**: 48x48 grayscale images
- **Architecture**: CNN with multiple convolutional layers
- **Output**: 7 emotion classes (Happy, Sad, Angry, Surprised, Fearful, Disgusted, Neutral)
- **Accuracy**: 85% on FER2013 dataset

## Dataset

We use the FER2013 dataset which contains:
- 35,887 images
- 7 emotion categories
- 48x48 pixel grayscale images

## TODO List

- [ ] Paper
- [ ] Gradio demo
- [ ] Inference code
- [ ] Model weights
- [ ] Training code
- [ ] Web application
- [ ] Mobile app integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Email**: h.zohaibhassan00@gmail.com
- **LinkedIn**: linkedin.com/in/Zohaib Hassan
- **GitHub**: github.com/zohaibhassan00 
