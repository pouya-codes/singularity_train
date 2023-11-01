import train.parser as parser
from train import *

if __name__ == "__main__":
    config = parser.get_args()
    mt = ModelTrainer(config)
    mt.run()
