#extractGRL.py
import os, json
import numpy as np
import torch
from dataset.CISDataset import RxAgnosticCISDataset
from model.resnet import ResNet
from model.GRL_B import ExtractorWithGRL  # same wrapper used in training

def extract_features(
    ckpt_dir = r"D:/午/EXTRACTOR/save/GRLGaps_0922_3dis_AF16",   # best_model.pt + label_maps.json
    data_root = r"D:/午/EXTRACTOR/dataset/0922_AF16_complex_low1MHz_linear_test/Q_matrix",
    save_root = r"D:/午/EXTRACTOR/RFF/0930_AF16_complex_low1MHz_linear_test",
    use_cuda = torch.cuda.is_available()
):
    device = torch.device("cuda:0" if use_cuda else "cpu")
    os.makedirs(save_root, exist_ok=True)

    print("=== Setup ===")
    print(f"ckpt_dir : {ckpt_dir}")
    print(f"data_root: {data_root}")
    print(f"save_root: {save_root}")
    print(f"device   : {device}")

    # --- load label_maps to recover NUM_DOMAINS (gaps) ---
    maps_path = os.path.join(ckpt_dir, "label_maps.json")
    with open(maps_path, "r", encoding="utf-8") as f:
        label_maps = json.load(f)
    num_domains = len(label_maps["dom_to_idx"])
    print(f"Loaded label_maps. num_gaps={num_domains}")

    # --- build the model exactly like training, load weights, eval mode ---
    extractor = ResNet(img_channels=1)  # same args as during training
    model = ExtractorWithGRL(extractor, num_domains=num_domains, grl_alpha=0.0)
    state = torch.load(os.path.join(ckpt_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print("Model loaded and set to eval().")

    # --- dataset to extract from ---
    ds = RxAgnosticCISDataset(data_root=data_root)
    total = len(ds)
    print(f"Dataset ready. num_items={total}")
    print("=== Begin extraction ===")

    # Streaming save like your original script (assumes ds is grouped by (gap, label))
    previous_label = None
    previous_gap   = None
    RFF = []   # will collect embeddings for current (gap,label)
    num = 1

    with torch.no_grad():
        for i in range(total):
            CIS, label, gap = ds[i]     # gap e.g. "0.1cm"
            x = CIS.unsqueeze(0).to(device)   # [1,1,H,W] (or [1,C,H,W])

            emb, _ = model(x, alpha=0.0)      # GRL scale irrelevant at eval
            emb_np = emb.squeeze(0).cpu().numpy()

            # boundary: new (gap,label) group starts
            if (previous_label != label) or (previous_gap != gap):
                # save the previous group first
                if (previous_label is not None) and len(RFF) > 0:
                    saved_folder_path = os.path.join(save_root, str(previous_gap))
                    os.makedirs(saved_folder_path, exist_ok=True)
                    arr = np.stack(RFF, axis=0)
                    out_path = os.path.join(saved_folder_path, f"{previous_label}.npy")
                    np.save(out_path, arr)
                    print(f"Saving card {previous_label} in {previous_gap}  -> shape {arr.shape}")

                # reset counters for new group
                num = 1
                print(f"Extracting card {label} in {gap}")
                RFF = []

            # collect current embedding
            RFF.append(emb_np)
            num += 1
            previous_label = label
            previous_gap   = gap

        # save the last group
        if len(RFF) > 0:
            saved_folder_path = os.path.join(save_root, str(previous_gap))
            os.makedirs(saved_folder_path, exist_ok=True)
            arr = np.stack(RFF, axis=0)
            out_path = os.path.join(saved_folder_path, f"{previous_label}.npy")
            np.save(out_path, arr)
            print(f"Saving card {previous_label} in {previous_gap}  -> shape {arr.shape}")

    print("=== Done ===")

if __name__ == "__main__":
    extract_features()
