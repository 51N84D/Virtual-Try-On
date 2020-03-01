from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.dataloader import *

if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")
    print("Command options are 'first_item' or 'first_batch'")

    opt = get_opt() 
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d' \
            % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed; embed()
