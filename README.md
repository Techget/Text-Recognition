## Optical Character Recognition engine with stroke width transform(swt), maximally stable extremal regions, connvolutional neural networks, and various morphological operations

### Dependencies required are
numpy==1.14.3,
opencv-python==3.4.0.12,
Pillow==5.1.0,
pypillowfight==0.2.4,
autocorrect,
spellchecker,
editdistance,
tensorflow,

### Usage
To run the OCR engine, use following command
```
python3 main.py demo.png demoGroundTruthText.txt
```
After running this, you'll see result.txt, with extracted texts, and coordinates and width&height of the text region, you'll also get textBlockdemo.png which is an image with bounding box indicating which part has been extracted.
To compare with pytesseract using:
```
python3 comparePytesseract.py demo.png demoGroundTruthText.txt
```

### MISC
To get a better understanding, have a look in the [report](https://docs.google.com/document/d/17LA3ertSA5a36B1Gfju-eBpXZ0ikf46P-n1piiYW97w/edit?usp=sharing)

We use 'char74k' to train the CNN model.


