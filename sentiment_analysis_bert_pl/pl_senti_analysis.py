from absl import app, flags, logging
import torch
import pytorch_lightning as pl
import transformers

FLAGS = flags.FLAGS

def main(_):
    logging.info("hello")

if __name__ == "__main__":
    app.run(main)