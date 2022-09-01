## Synthic

In this project we create a custom Gameboy emulator and ML pipeline for
synthesizing novel Gameboy music. We create a new encoding to represent Gameboy
music which represents the songs as streams of instructions and cycle timings
to the Gameboy audio hardware. We then train an autoregressive language model
to predict the next instruction and instruction time from a sequence of
previous instructions.

For this project we instrument the Mimic emulator, introducing a headless mode
which randomly explores games at more than 50x real-time. The modeling and
prediction work uses a Torch based Python code base and the remaining utilities
are written in Rust.

**TLDR;** Visit [this samples site](https://dev.parsed.dev/wav_samples/) for to
listen to a set of computer generated Gameboy songs.

### Overview 

The project is split into four individual tools that make up a data collection,
training, and synthesis pipeline.

#### Emulator

We modify the Mimic Gameboy emulator to support a fast playback mode (headless)
which can run games in parallel at 50x speed on my personal machine.  The
modified emulator has an instrumented memory bus that detects changes to any of
the Gameboy audio hardware and outputs them in a text based format. Our new
headless mode drivers games forward to discover new music by randomly pressing
buttons for an interval and then recording the period afterwards. We do not
record the period where buttons are being pressed to reduce the number of sound
effects that leak into samples.

#### Pruner

Our data pruner is a small utility program that selects subsequences of
emulator audio recordings for use as training data. Here, we discard samples
that don't mean heuristic quality controls (For example, where amplitude is
consistently low) or that we heuristically decide is not music and then take a
spread of remaining samples from throughout the emulation.

#### Predictor

Once we have collected and pruned our data we use the Predictor program to both
train a new model and synthesize novel audio from that model.  The predictor is
written in Python using torch and has two principle modes of operation,
training or generate. In training mode the predictor will take the audio
samples that we have generated and use them to train a model using the model
architecture specified in model.py. In generate mode the predictor will use a
model of the same architecture to predict new samples from a randomly selected
seed and output them to a file for playback.

#### Convertor

Until now, our representation of all of our audio samples has been in a custom
format that captures the instructions sent to audio hardware and their times.
To enjoy our synthesized music we now need to convert it into an audio wave and
write it to a file. Our final utility, convert2wav does exactly this, taking
any file encoded in our sample format and writing a .wav file as output.

#### Audio Format

Gameboy audio recordings are represented as 3-tuples of instruction, cycles
since the last Instruction, and channel. The possible types of Instruction are:

- **LSB** with params frequency: Set the least significant byte of the output
  channels frequency.
- **MSB** with parameters trigger, length enable, frequency: Set the most
  significant bit of a channels frequency along with trigger and length enable
flags.
- **VOL** with params volume, add, period: Set a channels volume and period
  registers along with an add bit.
- **DUTY** with params duty, length_load: Set the duty cycle and length load
  registers.

These are saved either as text or binary files. The text files are line
separated instructions in the form `CH <ch> <instruction> <params> AT <time>`
The binary files have a fixed 7 byte instruction size (2 bytes for the time
interval, one byte for channel, one byte for Instruction, and 3 for instruction
parameters).


#### Model Architecture

We model our stream of instructions as text and use a language model to predict
new outputs. 

### Usage

In this section we detail the process of collecting usable training data,
training a new model, and then using that model to generate new music using the
trained model.

**TLDR;** `./scripts/play_and_record_all <roms dir> && ./scripts/prune_all &&
python predictor/src/Predictor.py --mode split_data --source-dir ./pruned/
--training-data ~/scratch/training-data/ --test-data ~/scratch/test_data/
--model-dir ./local.model/ && python predictor/src/Predictor.py --mode fresh
--training-data ~/scratch/training-data --test-data ~/scratch/test_data
--model-dir ./local.model/ --output-path /tmp/ && ./scripts/generate_in_a_loop` 

#### Training data collection

We collected a large corpus of Gameboy ROM files from the internet archive. We
then let those ROMs run in an emulator each for ten minutes (real-time). To
repeat this a user can run `./scripts/play_and_record_all <roms dir>`. Once
execution finished we run the Pruner on the collected to collect a series of
short musical samples out of the recording from intervals where the emulator
was not pressing buttons. To repeat this a user can run `./scripts/prune_all`.
After this, our data is ready for model training. After this we can split the
data into a training and testing set by running: `python
predictor/src/Predictor.py --mode split_data --source-dir ./pruned/
--training-data ~/scratch/training-data/ --test-data ~/scratch/test_data/
--model-dir ./local.model/`

**TLDR;** `./scripts/play_and_record_all <roms dir> && ./scripts/prune_all &&
python predictor/src/Predictor.py --mode split_data --source-dir ./pruned/
--training-data ~/scratch/training-data/ --test-data ~/scratch/test_data/
--model-dir ./local.model/`

#### Model Training

Now that we have prepared some data, model training is straightforward. Simply
run `python predictor/src/Predictor.py --mode fresh --training-data
~/scratch/training-data --test-data ~/scratch/test_data --model-dir
./local.model/ --output-path /tmp/` and wait for the training to terminate,
which will happen automatically when testing loss stops going down. If you need
to shut down the program for whatever reason the program can be executed with
`--mode train` instead of `--mode fresh` to continue model training from the
last epoch.

**TLDR;** `python predictor/src/Predictor.py --mode fresh --training-data
~/scratch/training-data --test-data ~/scratch/test_data --model-dir
./local.model/ --output-path /tmp/`

#### Music Sample Generation

To generate individual samples using the model we have trained we can run run
`python predictor/src/Predictor.py --mode generate --training-data ./fast-test
--test-data ./fast-test --model-dir ./local.model/ --output-path /tmp/`. This
will place the output file at `/tmp/output.txt`. We can then play this file
back by running `./scripts/playback /tmp/output.txt`. To make this more
convenient the script `./scripts/generate_in_a_loop` can be run to generate
lots of random songs by seeding the generator with a random sample from
something in our testing data set.

**TLDR;** `./scripts/generate_in_a_loop`

#### Converting recordings to WAV files

Once we have generated some audio samples we need to convert from them into
audio files. The audio converter will automatically discard obviously bad
samples and outputs in the wav format. To invoke the converter run
`./scripts/convert_to_wav <sample_file> <output_dir>`. We also include the
script `./scripts/convert_all_to_wav` to convert all samples in a directory to
wav files.

**TLDR;** `./scripts/convert_all_to_wav  ./samples /tmp/out`
