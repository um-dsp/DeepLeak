import os
import torch
from torch_geometric.data import InMemoryDataset, Data

class DynamicGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DynamicGraphDataset, self).__init__(root, transform, pre_transform)
        self.data_list = []
        self.label_list = []
        self.class_list = []

    def add_attribut(self, graph_data, label, i_class):
        self.data_list.append(graph_data)
        self.label_list.append(label)
        self.class_list.append(i_class)

    def save(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((self.data_list, self.label_list, self.class_list), self.processed_paths[0])

    def load(self):
        data, labels, class_list = torch.load(self.processed_paths[0])
        self.data_list = data
        self.label_list = labels
        self.class_list = class_list

    def process(self):
        pass  # Not used in this dynamic setup

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx], self.label_list[idx], self.class_list[idx]
def load_data(save_path):
    dataset= DynamicGraphDataset(root=save_path)
    dataset.load()
    dataset=[(dataset[i][0].x,torch.tensor(dataset[i][1]),torch.tensor(dataset[i][2])) for  i in range (len(dataset))]
    return(dataset)