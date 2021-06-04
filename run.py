import model
import argparse

if __name__ == "__main__":

    print("Running model: press Ctrl+C to stop")

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_num", default=1, required=True, type=int)
    parser.add_argument("--epoch", default=1, required=True, type=int)

    args = parser.parse_args()
    
    model.run_model(args)
