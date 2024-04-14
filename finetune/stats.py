from torch.utils.data import Dataset
from nuimages import NuImages
import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, default="./data")
parser.add_argument("--version", type=str, default="v1.0-train")
args = parser.parse_args()


class NuDataset(Dataset):
    def __init__(self, path, version, verbose = False, transform = None):
        self.path = path
        self.version = version
        self.transform = transform
        self.nuim = NuImages(
            dataroot = path,
            version = version, 
            verbose = verbose,
            lazy = True
        )

    def __len__(self):
        return len(self.nuim.sample)
    
    def __getitem__(self, idx):
        sample = self.nuim.sample[idx]
        return sample
        

def count_frames(dataset, categories):
    nuim = dataset.nuim
    obj_tokens = []
    empty_tokens = []
    all_tokens = []
    categories_counter = {}
    pbar = tqdm.tqdm(total = len(nuim.object_ann) + len(nuim.surface_ann), desc = "Counting frames")

    for ann in nuim.object_ann:
        category = nuim.get('category', ann['category_token'])['name']
        token = ann['sample_data_token']
        if token not in all_tokens:
            all_tokens.append(token)

        if any(c in category for c in categories) and token not in obj_tokens:
            obj_tokens.append(token)
            if category not in categories_counter:
                categories_counter[category] = 1
            else:
                categories_counter[category] += 1

        pbar.update(1)

    for ann in nuim.surface_ann:
        category = nuim.get('category', ann['category_token'])['name']
        token = ann['sample_data_token']
        if token not in all_tokens:
            all_tokens.append(token)
        pbar.update(1)
    pbar.close()

    empty_tokens = [t for t in all_tokens if t not in obj_tokens]
    categories_counter["background"] = len(empty_tokens)

    macro_counter = {c: 0 for c in categories}
    for c in categories:
       for k in categories_counter.keys():
           if c in k:
               macro_counter[c] += categories_counter[k]
    macro_counter["background"] = categories_counter["background"]

    print(f"Found {len(obj_tokens)} positive frames")
    print(f"Found {len(empty_tokens)} empty frames")
    print(f"Total frames: {len(dataset)}")
    print(f"Total tokens: {len(all_tokens)}")
    return obj_tokens, empty_tokens, all_tokens, categories_counter, macro_counter


def get_data_info(dataset, categories):
    nuim = dataset.nuim
    pbar = tqdm.tqdm(total = len(nuim.sample), desc = "Getting data info")
    all_obj_names = []
    counter = {
        "safe": 0,
        "dangerous": 0
    }
    for s in nuim.sample:
        sd_token = s["key_camera_token"]
        obj_annots = [o for o in nuim.object_ann if o["sample_data_token"] == sd_token]
        obj_names = [nuim.get("category", o["category_token"])["name"] for o in obj_annots]
        cat_detected = {c: False for c in categories}
        for c in categories:
            if any(c in o for o in obj_names):
                cat_detected[c] = True

        if all(cat_detected.values()):
            counter["dangerous"] += 1
        else:
            counter["safe"] += 1
        
        pbar.update(1)
    pbar.close()
    # cat_counter = count_unique_targets(all_cat_names, categories)
    return counter


def count_unique_targets(obj_annots, ref_categories):
    counter = {c: 0 for c in ref_categories}
    counter["safe"] = 0
    counter["dangerous"] = 0
    for ann in obj_annots:
        bool_counter = {c: False for c in ref_categories}
        for rc in ref_categories:
            if (all (rc in a for a in ann)) and not bool_counter[rc]:
                counter[rc] += 1
                bool_counter[rc] = True
        if not any(bool_counter.values()):
            counter["safe"] += 1

    counter["dangerous"] = len(obj_annots) - counter["safe"]
    return counter


def plot_stats(counter):
    fig, ax = plt.subplots(1, 1, figsize = (12, 3))
    ax.set_xscale("log")
    ax.grid(which="major", axis="x")

    x = list(counter.keys())
    y = list(counter.values())

    barlist = ax.barh(x, y)
    for i, c in enumerate(x):
        ax.text(y[i], i, str(y[i]), va = "center")
        if c == "safe" or c == "dangerous":
            barlist[i].set_color("darkred")
        else:
            barlist[i].set_color("darkblue")

    fig.tight_layout()
    # fig.savefig(f"{args.version}.png", dpi = 600)


if __name__ == "__main__":
    dataset = NuDataset(
        path = args.data_path,
        version = args.version,
        transform = None,
        verbose = False
    )
    categories = [
        "human",
        "vehicle"
    ]

    print([c["name"] for c in dataset.nuim.])
    exit()

    counter = get_data_info(dataset, categories)
    print(counter)
    # plot_stats(counter)
    # plt.show()