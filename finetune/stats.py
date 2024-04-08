from torch.utils.data import Dataset
from nuimages import NuImages
import tqdm
import matplotlib.pyplot as plt
from collections import Counter


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
    counter = []
    for s in nuim.sample:
        sd_token = s["key_camera_token"]
        obj_annots = [o for o in nuim.object_ann if o["sample_data_token"] == sd_token]
        cat_names = [nuim.get("category", o["category_token"])["name"] for o in obj_annots]
        if not cat_names:
            cat_names = ["background"]
        tmp = []
        for cn in cat_names:
            for ct in categories:
                if (ct in cn) or (cn == "background"):
                    tmp.append(cn)
        counter.append(list(set(tmp)))
        pbar.update(1)
    pbar.close()
    return counter


def plot_stats(count_targets, dang_counter):
    fig, ax = plt.subplots(1, 1, figsize = (14, 8))
    ax.set_xscale("log")
    ax.grid(which="major", axis="x")

    x = list(count_targets.keys())
    x.extend(list(dang_counter.keys()))
    y = list(count_targets.values())
    y.extend(list(dang_counter.values()))

    barlist = ax.barh(x, y)
    for i, c in enumerate(x):
        ax.text(y[i], i, str(y[i]), va = "center")
        if c in count_targets:
            barlist[i].set_color("darkblue")
        else:
            barlist[i].set_color("darkred")

    fig.tight_layout()
    fig.savefig("stats_mini.png", dpi = 600)


if __name__ == "__main__":
    dataset = NuDataset(
        path = "/home/alberto/datasets/nuimages/",
        version = "v1.0-mini",
        transform = None,
        verbose = False
    )
    categories = [
        "human",
        "vehicle",
        "static_object.bicycle_rack",
    ]
    dang_counter = {
        "safe": 0,
        "dangerous": 0
    }
    counter = get_data_info(dataset, categories)
    for c in counter:
        if c == ["background"]:
            dang_counter["safe"] += 1
        else:
            dang_counter["dangerous"] += 1
    flat_counter = [
        c
        for sublist in counter
        for c in sublist
    ]
    count_targets = dict(Counter(flat_counter))
    plot_stats(count_targets, dang_counter)
    plt.show()