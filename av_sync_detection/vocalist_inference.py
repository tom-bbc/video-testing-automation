import torch
from vocalist.models.model import SyncTransformer
from vocalist.test_lrs2 import Dataset
from torch.utils import data as data_utils, eval_model

cpk = torch.load("vocalist/vocalist_weights/vocalist_5f_lrs2.pth", map_location='cpu')
model = SyncTransformer()
model.load_state_dict(cpk["state_dict"])


if __name__ == "__main__":
    # Params
    device = "mps"
    BATCH_SIZE = 1

    # Prepare input data

    # Split data into wav and jpg files per video,
    # with the path of each file from the data_root listed in the filename_list
    data_root_path = 'vocalist/prepared_data/'
    filename_list_path = 'inference'

    # Model
    checkpoint = torch.load("vocalist/vocalist_weights/vocalist_5f_lrs2.pth", map_location=torch.device(device))
    model = SyncTransformer(device=device).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    # Dataset and Dataloader setup
    test_dataset = Dataset('test')
    test_data_loader = data_utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)

    with torch.no_grad():
        eval_model(test_data_loader, device, model)
