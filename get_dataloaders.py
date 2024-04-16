from torch.utils.data import Dataset, DataLoader

def get_dataloaders(args, X_miss_train, Z_miss_train, X_miss_val, Z_miss_val, X_miss_test, Z_miss_test):
    train_dataset = MVIDataset(X_miss_train, Z_miss_train)
    val_dataset = MVIDataset(X_miss_val, Z_miss_val)
    test_dataset = MVIDataset(X_miss_test, Z_miss_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


class MVIDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]