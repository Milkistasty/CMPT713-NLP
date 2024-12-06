1. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

2. Install additional packages for training cases
```Shell
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### To run pretrained model:

3. Download these folders: [drivelink](https://drive.google.com/drive/folders/1XUOED7qq13gr3j8FtI8jrsZULMWzE0X0?usp=sharing)

4. Place `llava_lora_fn_cls2_4cls.tar.gz` in the `downloads/checkpoints/` directory: 

5. Place `dataset.tar.gz` in the `source` directory: 

5. Run and Evaluate Model. Generates output files containing Predictions and Ground truths. The evluation metrics will be displayed in the terminal. From the `source` directory:
```Shell
tar -xvzf dataset.tar.gz
tar -xvzf downloads/checkpoints/llava_lora_fn_cls2_4cls.tar.gz -C downloads/checkpoints/
sh infer0.sh
```

### To Train Model:
From the `source` directory:
```Shell
sh ft0.sh
```