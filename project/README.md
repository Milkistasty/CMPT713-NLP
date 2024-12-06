### Content Highlighs:
`source` directory contains the write-up, the ouput.txt file, and an eval.py file.

1. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip 
pip install -e .
```

2. Install additional packages for training cases
```Shell
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
### To evaluate pre-generated outputs:
```Shell
python3 eval.py
```

### To run pretrained model:
1. Download the contents of this: [drivelink](https://drive.google.com/drive/folders/1XUOED7qq13gr3j8FtI8jrsZULMWzE0X0?usp=sharing)

2. Place the `llava_lora_fn_cls2_4cls.tar.gz` into `download/checkpoints/`

3. Place the `dataset.tar.gz` into the `source` directory.

4. Run and Evaluate Model. Generates output output files containing Predictions and Ground truths. The evluation metrics will be displayed in the terminal.
```Shell
tar -xvzf dataset.tar.gz
tar -xvzf downloads/checkpoints/llava_lora_fn_cls2_4cls.tar.gz -C downloads/checkpoints/
sh infer0.sh
```

### To train the model
1. Download the contents of this: [drivelink](https://drive.google.com/drive/folders/1XUOED7qq13gr3j8FtI8jrsZULMWzE0X0?usp=sharing)

2. Place the `dataset.tar.gz` into the `source` directory.

3. Train the model
```Shell
tar -xvzf dataset.tar.gz
sh ft0.sh
```

