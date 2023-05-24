import os 
from pathlib import Path 
import pickle
import warnings

import torch
import numpy as np
from torch.hub import download_url_to_file
from torch.utils.data import Dataset
from torchaudio.datasets.utils import _extract_tar

import librosa
import time
from tqdm import tqdm

class CheckThisCell(Exception):
    pass

class Kalaka(Dataset):
    RELEASE_CONFIGS = {'train100': {'url': 'http://groups.tecnico.ulisboa.pt/speechproc/pf23/lab2/train100.tar.gz' , 'checksum':'fc5a0b17c142ee5a94d6ed9dddb249d9725d0ed5f07225ca72bc96e5541e2211'},
                'train': {'url': 'http://groups.tecnico.ulisboa.pt/speechproc/pf23/lab2/train.tar.gz' , 'checksum':'bfd0ae025c78e72f9a58702ed440983e023ab0220ef9829306b97b89c0d0a164'},
                'dev': {'url': 'http://groups.tecnico.ulisboa.pt/speechproc/pf23/lab2/dev.tar.gz' , 'checksum':'32c657beeb16831553c41c63aa952892b821d48ae6cfdcfeedca08ae62ca2da2'},
                'evl': {'url': 'http://groups.tecnico.ulisboa.pt/speechproc/pf23/lab2/evl.tar.gz' , 'checksum':'8db55865845b25a5069a1206782fdfacc9ab76de6bda253b4184cf59a91107bc'}
                }
                   
    def __init__(self, root : str, dataset_id: str, transform_id: str = "feat", audio_transform : callable = None, chunk_size : int = -1, chunk_hop : int = -1, chunk_transform : callable = None) -> None:
        
        if dataset_id not in Kalaka.RELEASE_CONFIGS:
            raise ValueError("Not known data set in Kalaka")
        
        if audio_transform is None:
            raise ValueError("Need to define some tranformation from audiofile to features")

        self.path = Path(root) / dataset_id
        self.url = Kalaka.RELEASE_CONFIGS[dataset_id]['url']

        self.archive = os.path.basename(self.url)
        self.archive = Path(root) / self.archive

        self.audio_dir = self.path / 'audio'
        self.feat_dir = self.path / transform_id
        self.key_file = self.path / 'key.lst'

        self.audio_transform = audio_transform
        self.chunk_size = chunk_size
        self.chunk_hop = chunk_hop
        self.chunk_transform = chunk_transform

        if self.chunk_size > 0 and self.chunk_hop <= 0:
            self.chunk_hop = self.chunk_size

        self.download_data(Kalaka.RELEASE_CONFIGS[dataset_id]['checksum'])
        self.data_to_feat()

        if not os.path.isfile(self.key_file):
            raise RuntimeError("Key file does not exist. There was some problem downloading data.")

        if not os.path.isdir(self.feat_dir):
            raise RuntimeError("Features directory does not exist. There was some problem applying feature extraction.")
    
        self._walker = []
        with open(self.key_file) as file:
            for line in file:
                basename, _, label_num = line.split()[0].strip(), line.split()[1].strip(), int(line.split()[2].strip())
                self._walker.extend([(c, basename, label_num) for c in os.listdir(self.feat_dir / basename )])
    
    def __getitem__(self, index):
        featIn, basename, label_num = self._walker[index]

        feats = pickle.load(open(self.feat_dir / basename / featIn, 'rb'))
        
        return feats, label_num, basename
    
    def __len__(self):
        return len(self._walker)
                   
    def download_data(self,  checksum : str = None) -> None:   
        if not os.path.isdir(self.path):
            if not os.path.isfile(self.archive):
                download_url_to_file(self.url, self.archive, hash_prefix=checksum)
            _extract_tar(self.archive)

    def data_to_feat(self) -> None:
        
        if not os.path.isfile(self.key_file):
            raise RuntimeError("Key file does not exist. Please download it first. There was some problem during data download and extreaction.")

        if os.path.isdir(self.feat_dir):
            warnings.warn("The feature directory already exists, and no new feature extraction will be performed.")
        else:
            if not os.path.isdir(self.audio_dir):
                raise RuntimeError("Audios directory does not exist. There was some problem during data download and extreaction.")
   
            ## Read number of lines
            os.mkdir(self.feat_dir)
            # with open(self.key_file) as file:
            #     num_files = len(file.readlines())

            # Extract features
            with open(self.key_file) as file:
                for line in tqdm(file.readlines()):
                    basename, _, _ = line.split()[0].strip(), line.split()[1].strip(), int(line.split()[2].strip())
                    audioin = self.audio_dir / f'{basename}.wav'
                    featOutPath = self.feat_dir / f'{basename}' 
                    # labeldev.append(line.split()[1].strip())

                    if not os.path.isdir(featOutPath):
                        os.mkdir(featOutPath)
                    
                    # print(f'\t{audioin}...', end='')
                    feats = self.audio_transform(str(audioin)) # The output of this is (Ntime x Dim)

                    finish = False
                    if self.chunk_size > 0:
                        for b in range(0, feats.shape[0], self.chunk_hop-1):
                            this_feats = feats[b:b+self.chunk_size]
                            if this_feats.shape[0] < self.chunk_size:
                                this_feats = np.concatenate((this_feats, np.zeros((self.chunk_size-this_feats.shape[0], this_feats.shape[1]), dtype=this_feats.dtype)))
                                finish = True
                            if self.chunk_transform is not None:
                                this_feats = self.chunk_transform(this_feats)

                            pickle.dump(this_feats, open(featOutPath / f'{basename}.{b//self.chunk_hop}.feat' , 'wb'))
                            if finish:
                                break 
                    else:
                        pickle.dump(feats, open(featOutPath / f'{basename}.feat' , 'wb'))
                            
                    # print(f'[{i+1}/{num_files}]')

