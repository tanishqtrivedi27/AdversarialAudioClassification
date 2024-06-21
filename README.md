# AdversarialAudioClassification

The project deals with conducting thorough experiments to investigate the influence of adversarial attacks on deep ESC models. These experiments encompass three standard ESC datasets, four deep neural networks, and four distinct adversarial attack methods.

We also examine transferability of adversarial attacks, both class-aware and unaware, across different deep models considering both targeted and untargeted attack scenarios.

```
python data_setup.py --annotations_file ANNOTATIONS --audio-dir DIR --sample_rate 44100
```
```
python train.py --dataset_name DATASET --model MODEL --batch_size BATCHES --lr LR --num_epochs EPOCHS --patience PATIENCE --num_classes NUM_CLASSES
```

### Slides
[Slides](https://drive.google.com/file/d/1M7Tqll-uw1JxA4CziRJ59d7gZkiMCcGT/view?usp=sharing)

### Results
[Results](https://docs.google.com/spreadsheets/d/1XqUkbvIkWKFGIn7iM9qxf_aic262oZAPmHTzMqznIqY/edit?usp=sharing)