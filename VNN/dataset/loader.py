"""Datasets loading and preprocessing"""

import inspect
import numpy as np
import os 
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
    

def normalize_cols(df, cols_to_normalize, feature_range=(0,1)):
    """ Rescale column features to a range of [x,y]"""
    # Create scaler object
    scaler = MinMaxScaler(feature_range=feature_range)
    # Normalize columns
    df_norm = pd.DataFrame(scaler.fit_transform(df[cols_to_normalize]), columns=cols_to_normalize)
    
    return df_norm


def get_cols_dummies(df, cols_to_dummies, feature_range):
    """Convert columns into indicator variables"""
    df_dummies = pd.DataFrame()
    for col in cols_to_dummies:
        df_dummies = df_dummies.append(pd.get_dummies(df[col], prefix=col, dtype='float64'))
    
    df_dummies = df_dummies.replace((0,1), feature_range)
    
    return df_dummies

 
def split_data(x, y, random_seed, test_ratio):
    """Split data"""
    if test_ratio > 0:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, 
            test_size=test_ratio, 
            random_state=random_seed, 
            shuffle=True
        )
    else:
        x_train = x
        y_train = y
        x_test = x_train[0:0]
        y_test = y_train[0:0]
        
    
    return x_train, x_test, y_train, y_test
    
 
def load_wine(random_seed=1, test_ratio=0.2, feature_range=(0,1)):
    """Load and preprocess wine dataset"""
    # Read data from disk
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, 'sources/wine/wine.data'), names=[
        'alcohol',
        'alic_acid',
        'ash',
        'alcalinity',
        'magnesium',
        'total_phenols',
        'flavanoids',
        'nonflavanoid_phenols',
        'proanthocyanins',
        'color_intensity',
        'hue',
        'OD280',
        'OD315',
        'proline'
    ])
    
    # Normalize columns
    cols_to_normalize = [
        'alic_acid',
        'ash',
        'alcalinity',
        'magnesium',
        'total_phenols',
        'flavanoids',
        'nonflavanoid_phenols',
        'proanthocyanins',
        'color_intensity',
        'hue',
        'OD280',
        'OD315',
        'proline'
    ]
    df_norm = normalize_cols(df, cols_to_normalize, feature_range)
    
    # Covert columns into indicator variables
    cols_to_dummies = ['alcohol']
    df_dummies = get_cols_dummies(df, cols_to_dummies, feature_range)
    
    # Split data
    x = df_norm.to_numpy()
    y = df_dummies.to_numpy()
    x_train, x_test, y_train, y_test = split_data(x, y, random_seed, test_ratio)
    
    return x_train, x_test, y_train, y_test
    

def load_sensorless(random_seed=1, test_ratio=0.2, feature_range=(0,1)):
    """Load and preprocess sensorless dataset"""
    # Read data from disk
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, 'sources/sensorless/Sensorless_drive_diagnosis.txt'), sep=' ', header=None)
    
    # Normalize columns
    cols_to_normalize = np.arange(48) # Columns 0-47
    df_norm = normalize_cols(df, cols_to_normalize, feature_range)
    
    # Covert columns into indicator variables
    cols_to_dummies = [48]
    df_dummies = get_cols_dummies(df, cols_to_dummies, feature_range)
    
    # Split data
    x = df_norm.to_numpy()
    y = df_dummies.to_numpy()
    x_train, x_test, y_train, y_test = split_data(x, y, random_seed, test_ratio)
    
    return x_train, x_test, y_train, y_test
    

def load_mnist(random_seed=1, test_ratio=0.2, feature_range=(0,1)):
    """Load and process MNIST dataset"""
    # Load data from repository
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Preprocess training set
    x_train = (x_train.astype('float64') / 255) * (feature_range[1] - feature_range[0]) + feature_range[0]
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = pd.get_dummies(y_train, dtype='float64').replace((0,1), feature_range).to_numpy()
    
    # Preprocess testing set
    x_test = (x_test.astype('float64') / 255) * (feature_range[1] - feature_range[0]) + feature_range[0]
    x_test = np.expand_dims(x_test, axis=-1)
    y_test = pd.get_dummies(y_test, dtype='float64').replace((0,1), feature_range).to_numpy()
    
    return x_train, x_test, y_train, y_test


def load_yacht_hydrodynamics(random_seed=1, test_ratio=0.2, feature_range=(0,1)):
    """Load yacht hydrodynamics dataset"""
    # Read data from disk
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, 'sources/yacht_hydrodynamics/yacht_hydrodynamics.data'), sep=' ', header=None, names=[
        'longitudinal_position',
        'prismatic_coefficient',
        'length-displacement_ratio',
        'beam-draugh_ratio',
        'length-beam_ratio',
        'froude_number',
        'residuary_resistance'
    ])
    
    # Normalize columns
    cols_normalize = [
        'longitudinal_position',
        'prismatic_coefficient',
        'length-displacement_ratio',
        'beam-draugh_ratio',
        'length-beam_ratio',
        'froude_number',
        'residuary_resistance'
    ]
    df_norm = normalize_cols(df, cols_normalize, feature_range)
    
    # Get input and output data
    input_cols = [
        'longitudinal_position',
        'prismatic_coefficient',
        'length-displacement_ratio',
        'beam-draugh_ratio',
        'length-beam_ratio',
        'froude_number'
    ]
    x = df_norm[input_cols].to_numpy()
    output_cols = [
        'residuary_resistance'
    ]
    y = df_norm[output_cols].to_numpy()
    
    # Split data
    x_train, x_test, y_train, y_test = split_data(x, y, random_seed, test_ratio)
    
    return x_train, x_test, y_train, y_test


def load_cardiotocography1(random_seed=1, test_ratio=0.2, feature_range=(0,1)):
    """Load cardiotocography dataset for FHR pattern class prediction"""
    # Read data from disk
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, 'sources/cardiotocography/cardiotocography.csv'))
    
    # Normalize columns
    cols_to_normalize = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV',
       'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean',
       'Median', 'Variance', 'Tendency']
    df_norm = normalize_cols(df, cols_to_normalize, feature_range)
    
    # Covert columns into indicator variables
    cols_to_dummies = ['CLASS']
    df_dummies = get_cols_dummies(df, cols_to_dummies, feature_range)
    
    # Split data
    x = df_norm.to_numpy()
    y = df_dummies.to_numpy()
    x_train, x_test, y_train, y_test = split_data(x, y, random_seed, test_ratio)
    
    return x_train, x_test, y_train, y_test


def load_cardiotocography2(random_seed=1, test_ratio=0.2, feature_range=(0,1)):
    """Load cardiotocography dataset for fetal state class prediction"""
    # Read data from disk
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, 'sources/cardiotocography/cardiotocography.csv'))
    
    # Normalize columns
    cols_to_normalize = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV',
       'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean',
       'Median', 'Variance', 'Tendency']
    df_norm = normalize_cols(df, cols_to_normalize, feature_range)
    
    # Covert columns into indicator variables
    cols_to_dummies = ['NSP']
    df_dummies = get_cols_dummies(df, cols_to_dummies, feature_range)
    
    # Split data
    x = df_norm.to_numpy()
    y = df_dummies.to_numpy()
    x_train, x_test, y_train, y_test = split_data(x, y, random_seed, test_ratio)
    
    return x_train, x_test, y_train, y_test


def load_computer_hardware(random_seed=1, test_ratio=0.2, feature_range=(0,1)):
    """Load computer hardware dataset"""
    # Read data from disk
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, 'sources/computer_hardware/machine.data'), header=None, names=[
        'vendor',
        'model',
        'myct',
        'mmin',
        'mmax',
        'cach',
        'chmin',
        'chmax',
        'prp',
        'erp'
    ])

    # Normalize columns
    cols_normalize = [
        'myct',
        'mmin',
        'mmax',
        'cach',
        'chmin',
        'chmax',
        'prp'
    ]
    df_norm = normalize_cols(df, cols_normalize, feature_range)

    # Get input and output data
    input_cols = [
        'myct',
        'mmin',
        'mmax',
        'cach',
        'chmin',
        'chmax'
    ]
    x = df_norm[input_cols].to_numpy()
    output_cols = ['prp']
    y = df_norm[output_cols].to_numpy()

    # Split data
    x_train, x_test, y_train, y_test = split_data(x, y, random_seed, test_ratio)
    
    return x_train, x_test, y_train, y_test


def load_dry_bean(random_seed=1, test_ratio=0.2, feature_range=(0,1)):
    """Load dry bean dataset"""
    # Read data from disk
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, 'sources/dry_bean/dry_bean.csv'))

    # Normalize columns
    cols_to_normalize = [
        'Area',
        'Perimeter',
        'MajorAxisLength',
        'MinorAxisLength',
        'AspectRation',
        'Eccentricity',
        'ConvexArea',
        'EquivDiameter',
        'Extent',
        'Solidity',
        'roundness',
        'Compactness',
        'ShapeFactor1',
        'ShapeFactor2',
        'ShapeFactor3',
        'ShapeFactor4'
    ]
    df_norm = normalize_cols(df, cols_to_normalize, feature_range)
        
    # Covert columns into indicator variables
    cols_to_dummies = ['Class']
    df_dummies = get_cols_dummies(df, cols_to_dummies, feature_range)

    # Split data
    x = df_norm.to_numpy()
    y = df_dummies.to_numpy()
    x_train, x_test, y_train, y_test = split_data(x, y, random_seed, test_ratio)
    
    return x_train, x_test, y_train, y_test


def load_energy_efficiency(random_seed=1, test_ratio=0.2, feature_range=(0,1)):
    """Load energy efficiency dataset"""
    # Read data from disk
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, 'sources/energy_efficiency/energy_efficiency.csv'))

    # Normalize columns
    cols_to_normalize = [
        'X1',
        'X2',
        'X3',
        'X4',
        'X5',
        'X6',
        'X7',
        'X8',
        'Y1',
        'Y2'
    ]
    df_norm = normalize_cols(df, cols_to_normalize, feature_range)
        
    # Get input and output data
    input_cols = [
        'X1',
        'X2',
        'X3',
        'X4',
        'X5',
        'X6',
        'X7',
        'X8'
    ]
    x = df_norm[input_cols].to_numpy()
    output_cols = ['Y1','Y2']
    y = df_norm[output_cols].to_numpy()

    # Split data
    x_train, x_test, y_train, y_test = split_data(x, y, random_seed, test_ratio)
    
    return x_train, x_test, y_train, y_test


def load_kinematics(random_seed=1, test_ratio=0.2, feature_range=(0,1)):
    """Load dataset of kinematics of an 8 link robot arm"""
    # Read data from disk
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, 'sources/kinematics/kin8nm.data'), header=None, names=[
        'theta1',
        'theta2',
        'theta3',
        'theta4',
        'theta5',
        'theta6',
        'theta7',
        'theta8',
        'y'
    ])

    # Normalize columns
    cols_to_normalize = [
        'theta1',
        'theta2',
        'theta3',
        'theta4',
        'theta5',
        'theta6',
        'theta7',
        'theta8',
        'y'
    ]
    df_norm = normalize_cols(df, cols_to_normalize, feature_range)
        
    # Get input and output data
    input_cols = [
        'theta1',
        'theta2',
        'theta3',
        'theta4',
        'theta5',
        'theta6',
        'theta7',
        'theta8'
    ]
    x = df_norm[input_cols].to_numpy()
    output_cols = ['y']
    y = df_norm[output_cols].to_numpy()

    # Split data
    x_train, x_test, y_train, y_test = split_data(x, y, random_seed, test_ratio)
    
    return x_train, x_test, y_train, y_test


def load_winequality_red(random_seed=1, test_ratio=0.2, feature_range=(0,1)):
    """Load red wine quality dataset"""
    # Read data from disk
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, 'sources/winequality/winequality-red.csv'), sep=";")

    # Normalize columns
    cols_to_normalize = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality"
    ]
    df_norm = normalize_cols(df, cols_to_normalize, feature_range)
        
    # Get input and output data
    input_cols = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol"
    ]
    x = df_norm[input_cols].to_numpy()
    output_cols = ["quality"]
    y = df_norm[output_cols].to_numpy()

    # Split data
    x_train, x_test, y_train, y_test = split_data(x, y, random_seed, test_ratio)
    
    return x_train, x_test, y_train, y_test


def load_winequality_white(random_seed=1, test_ratio=0.2, feature_range=(0,1)):
    """Load red wine quality dataset"""
    # Read data from disk
    dir_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(dir_path, 'sources/winequality/winequality-white.csv'), sep=";")

    # Normalize columns
    cols_to_normalize = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality"
    ]
    df_norm = normalize_cols(df, cols_to_normalize, feature_range)
        
    # Get input and output data
    input_cols = [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol"
    ]
    x = df_norm[input_cols].to_numpy()
    output_cols = ["quality"]
    y = df_norm[output_cols].to_numpy()

    # Split data
    x_train, x_test, y_train, y_test = split_data(x, y, random_seed, test_ratio)
    
    return x_train, x_test, y_train, y_test


def load_cifar10(random_seed=1, test_ratio=0.2, feature_range=(0,1)):
    """Load and process MNIST dataset"""
    # Load data from repository
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Preprocess training set
    x_train = (x_train.astype('float64') / 255) * (feature_range[1] - feature_range[0]) + feature_range[0]
    y_train = pd.get_dummies(y_train.flatten(), dtype='float64').replace((0,1), feature_range).to_numpy()
    
    # Preprocess testing set
    x_test = (x_test.astype('float64') / 255) * (feature_range[1] - feature_range[0]) + feature_range[0]
    y_test = pd.get_dummies(y_test.flatten(), dtype='float64').replace((0,1), feature_range).to_numpy()
    
    return x_train, x_test, y_train, y_test


def load_fashion_mnist(random_seed=1, test_ratio=0.2, feature_range=(0,1)):
    """Load and process MNIST dataset"""
    # Load data from repository
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Preprocess training set
    x_train = (x_train.astype('float64') / 255) * (feature_range[1] - feature_range[0]) + feature_range[0]
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = pd.get_dummies(y_train, dtype='float64').replace((0,1), feature_range).to_numpy()
    
    # Preprocess testing set
    x_test = (x_test.astype('float64') / 255) * (feature_range[1] - feature_range[0]) + feature_range[0]
    x_test = np.expand_dims(x_test, axis=-1)
    y_test = pd.get_dummies(y_test, dtype='float64').replace((0,1), feature_range).to_numpy()
    
    return x_train, x_test, y_train, y_test


data_loaders = {
    'cardiotocography1': load_cardiotocography1,
    'cardiotocography2': load_cardiotocography2,
    'cifar10': load_cifar10,
    'computer_hardware': load_computer_hardware,
    'dry_bean': load_dry_bean,
    'energy_efficiency': load_energy_efficiency,
    'fashion_mnist': load_fashion_mnist,
    'kinematics': load_kinematics,
    'mnist': load_mnist,
    'sensorless': load_sensorless,
    'wine': load_wine,
    'winequality_red': load_winequality_red,
    'winequality_white': load_winequality_white,
    'yacht_hydrodynamics': load_yacht_hydrodynamics
}

shapes = {
    'cardiotocography1': ([21], [10]),
    'cardiotocography2': ([21], [3]),
    'cifar10': ([32, 32, 3], [10]),
    'computer_hardware': ([6], [1]),
    'dry_bean': ([16], [7]),
    'energy_efficiency': ([8], [2]),
    'fashion_mnist': ([28, 28, 1], [10]),
    'kinematics': ([8], [1]),
    'mnist': ([28, 28, 1], [10]),
    'sensorless': ([48], [11]),
    'wine': ([13], [3]),
    'winequality_red': ([11], [1]),
    'winequality_white': ([11], [1]),
    'yacht_hydrodynamics': ([6], [1])
}

def load(name, **kwargs):
    if name in data_loaders:
        return data_loaders[name](**kwargs)
    else:
        raise ValueError(f'Dataset "{name}" does not exists!')

def get_shapes(name):
    if name in shapes:
        return shapes[name]
    else:
        raise ValueError(f'Dataset "{name}" does not exists!')