## Prerequisites

- Required dependencies listed in `requirements.txt`

## Instructions

### Step 1: Environment Setup

1. Clone the repository and navigate to the project directory.
2. Set up a virtual environment and install dependencies:
    ```
    python3.10 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt
    ```
3. Activate the environment
    `source venv/bin/activate`

### Step 2: Train and Save the Model
1. Prepare Data for ORPO
    `python3 answer/data_processing.py`
1. Run the training script
    `python3 answer/train.py`

### Step 3: Obtain Results on dev.txt and test.txt
1. Generate the output files:
    `python3 zipout.py`
2. Check and validate the results:
    `python3 check.py`