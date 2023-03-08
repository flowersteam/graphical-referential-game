import torch, os
import numpy as np

from utils import device, path

def generate_systematic_dataset(n=10, m_list='all'):

    data_root = os.path.join(path,"data/datasets")
    if not os.path.exists(data_root):
        os.mkdir(data_root)
    data_path = os.path.join(data_root,"n"+str(n)+"_m"+ ("_all" if m_list == 'all' else "".join([str(i) for i in m_list])) + ".pt")

    if not os.path.exists(data_path):
        dataset = ((torch.arange(2**n).unsqueeze(1) >> torch.arange(n-1, -1, -1)) & 1).float()
        # Remove empty referent
        dataset = dataset[torch.sum(dataset,1)>0]

        # Only keep referents having m features such that m is in m_list
        for m in range(1,n+1):
            if m_list != 'all' and not (m in m_list):
                dataset = dataset[torch.sum(dataset,1)!=m]

        torch.save(dataset,data_path)

    return data_path


def separate_systematic_dataset(dataset, ratio=0.25):
    train_set, test_set = torch.empty(0,dataset.shape[1]), torch.empty(0,dataset.shape[1])
    # For compositionality evaluation, we always keep all 1-feature referents in train set
    train_set = dataset[torch.sum(dataset,1)==1]
    for m in torch.unique(torch.sum(dataset,1)):
        if m!=1:
            m_bin = dataset[torch.sum(dataset,1)==m]
            np.random.shuffle(m_bin.numpy())
            bin_train, bin_test = m_bin[0:math.ceil(m_bin.shape[0]*ratio)], m_bin[math.ceil(m_bin.shape[0]*ratio):]
            train_set = torch.cat((train_set,bin_train))
            test_set  = torch.cat((test_set,bin_test))
    return train_set, test_set
