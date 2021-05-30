"""Dataset preparation"""

from sklearn.model_selection import KFold
from tensorflow.data import Dataset
from .loader import load, get_shapes

def get_dataset_shapes(name):
    return get_shapes(name)

def get_test_datasets(name, random_seed=1, test_ratio=0.2, batch_size=32, feature_range=(0,1)):
    """Pack data into training and testing datasets"""
    x_train, x_test, y_train, y_test = load(
        name, 
        random_seed=random_seed,
        test_ratio=test_ratio,
        feature_range=feature_range
    )
        
    train_dataset = Dataset.from_tensor_slices((
        x_train, y_train
    ))
    test_dataset = Dataset.from_tensor_slices((
        x_test, y_test
    ))
    
    return train_dataset.batch(batch_size), test_dataset.batch(batch_size)

def get_validation_datasets(name, random_seed=1, test_ratio=0.2, k=5, batch_size=32, feature_range=(0,1)):
    """Generate datasets for k-fold crossvalidation"""
    x_train, _, y_train, _ = load(
        name, 
        random_seed=random_seed,
        test_ratio=test_ratio,
        feature_range=feature_range
    )
    
    k_fold = KFold(n_splits=k, random_state=None, shuffle=False)
    for train_index, valid_index in k_fold.split(x_train):
        train_dataset = Dataset.from_tensor_slices((
            x_train[train_index], y_train[train_index]
        ))
        valid_dataset = Dataset.from_tensor_slices((
            x_train[valid_index], y_train[valid_index]
        ))
        
        yield train_dataset.batch(batch_size), valid_dataset.batch(batch_size)