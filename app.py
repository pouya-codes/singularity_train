import train.parser as parser
from train import *

if __name__ == "__main__":
    config = parser.parse_args()
    mt = ModelTrainer(config)
    mt.run()
