# Self-Learning Robotic Arm

This project demonstrates how to create a self-learning robotic arm using reinforcement learning, which can be adapted for various tasks. The training is done on a computer, and the model is simplified for use on an Arduino.

## Setup

### Requirements

- Python 3.7+
- TensorFlow
- Gym
- Numpy
- Arduino

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/self-learning-robotic-arm.git
    cd self-learning-robotic-arm
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Train the reinforcement learning model:
    ```bash
    python train_model.py
    ```

4. Generate the lookup table for Arduino:
    ```bash
    python generate_lookup_table.py
    ```

5. Upload the Arduino sketch to your Arduino board:
    - Copy the `lookup_table.h` file to the Arduino sketch directory.
    - Open the Arduino IDE and create a new sketch.
    - Copy the provided Arduino code into the sketch.
    - Upload the sketch to your Arduino board.
