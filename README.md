# PyTorch Neural Turing Machines with Pretty Visualizer
[Neural Turing Machines, arxiv:1410.5401](https://arxiv.org/abs/1410.5401)

![](demo.gif)


### Requirements
- python3
- pytorch 0.4
<br>
<br>

### Usage
```
usage: train.py [-h] [--sequence_length SEQUENCE_LENGTH]
                [--sequence_width SEQUENCE_WIDTH]
                [--num_memory_locations NUM_MEMORY_LOCATIONS]
                [--memory_vector_size MEMORY_VECTOR_SIZE]
                [--training_size TRAINING_SIZE]
                [--controller_hidden_size CONTROLLER_HIDDEN_SIZE]
                [--controller_output_size CONTROLLER_OUTPUT_SIZE]
                [--learning_rate LEARNING_RATE] [--min_grad MIN_GRAD]
                [--max_grad MAX_GRAD] [--load LOAD] [--save SAVE]
                [--monitor_state]
```
<br>

### How to run the visualizer

```
cd path/to/this/project
python -m http.server 8000
```
and open another terminal

```
cd path/to/this/project
python train.py --monitor_state
```