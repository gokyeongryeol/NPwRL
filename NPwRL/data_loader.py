import numpy as np
import os
import random
from PIL import Image
import pdb
import itertools
import pickle
import torch 
import torch.nn.functional as F
from torch.distributions import Normal


class GaussianProcess(object):
    def __init__(self, is_train):
        self.is_train = is_train
    
    def generate_batch(self, batch_size, noise, is_test):
        num_context = np.random.randint(3, 47)
        num_target = np.random.randint(3, 50-num_context)
        
        length = np.random.uniform(0.1, 0.6, size=(batch_size,1,1,1))
        sigma = np.random.uniform(0.1, 1.0, size=(batch_size,1,1,1))
        
        if self.is_train:
            init_inputs = np.random.uniform(-2.0, 2.0, size=(batch_size, num_context+num_target, 1))
        else:
            init_inputs = np.linspace(-2.0, 2.0, num=1000)
            init_inputs = np.repeat(init_inputs.reshape(1, -1, 1), repeats=batch_size, axis=0)
            
        x1 = np.expand_dims(init_inputs, axis=1)
        x2 = np.expand_dims(init_inputs, axis=2)
        
        kernel = sigma**2 * np.exp(-0.5 * np.square(x1-x2) / length**2)
        kernel = np.sum(kernel, axis=-1)
        kernel += (0.02 ** 2) * np.identity(init_inputs.shape[1])
        
        cholesky = np.linalg.cholesky(kernel)
        outputs = np.matmul(cholesky, np.random.normal(size=(batch_size, init_inputs.shape[1], 1)))
        
        init_inputs = torch.from_numpy(init_inputs)
        outputs = torch.from_numpy(outputs)
        
        rng = np.random.default_rng()
        random_idx = rng.permutation(outputs.shape[1])

        noise_shape = (batch_size, num_target, 1)
        outputs[:, random_idx[-num_target:], :] += noise * torch.randn(noise_shape)
        
        Cx = init_inputs[:, random_idx[:num_context], :]
        Cy = outputs[:, random_idx[:num_context], :]
        if self.is_train and is_test:
            Tx = init_inputs[:, random_idx[-num_target:], :] 
            Ty = outputs[:, random_idx[-num_target:], :]
        else:
            Tx = init_inputs
            Ty = outputs
        return (Cx, Tx), (Cy, Ty)


class mini_tiered_ImageNet(object):
    def __init__(self, data_source, is_train):
        self.data_source = data_source
        self.is_train = is_train
        
        if is_train:
            self.metasplit = ['train', 'val']
        else:
            self.metasplit = ['test']
        
        self.construct_data()
    
    def construct_data(self):
        self.embeddings = {}
        for d in self.metasplit:
            with open(os.path.join('../data/'+self.data_source, d+'_embeddings.pkl'), 'rb') as file:
                self.embeddings[d] = pickle.load(file, encoding='latin1')
       
        self.image_by_class = {}
        self.embed_by_name = {}
        self.class_list = {}
        for d in self.metasplit:
            self.image_by_class[d] = {}
            self.embed_by_name[d] = {}
            self.class_list[d] = set()
            keys = self.embeddings[d]["keys"]
            for i, k in enumerate(keys):
                _, class_name, img_name = k.split('-')
                if (class_name not in self.image_by_class[d]):
                    self.image_by_class[d][class_name] = []
                self.image_by_class[d][class_name].append(img_name) 
                self.embed_by_name[d][img_name] = self.embeddings[d]["embeddings"][i]
                self.class_list[d].add(class_name)
            
            self.class_list[d] = list(self.class_list[d])

    def generate_batch(self, batch_size, num_context, num_target, num_classes, is_test=False):
        if self.is_train:
            if is_test:
                metasplit = 'val'
            else:
                metasplit = 'train'
        else:
            metasplit = 'test'

        batch = {'input':[], 'label':[]}
        for b in range(batch_size):
            shuffled_classes = self.class_list[metasplit].copy()
            random.shuffle(shuffled_classes)

            shuffled_classes = shuffled_classes[:num_classes]

            inp = [[] for i in range(num_classes)]
            lab = [[] for i in range(num_classes)]

            for c, class_name in enumerate(shuffled_classes):
                images = np.random.choice(self.image_by_class[metasplit][class_name], num_target)
                for i in range(num_target):
                    embed = self.embed_by_name[metasplit][images[i]]
                    inp[c].append(embed)
                    lab[c].append(c)

            permutations = list(itertools.permutations(range(num_classes)))
            order = random.choice(permutations)
            inputs = [inp[i] for i in order]
            labels = [lab[i] for i in order]

            batch['input'].append(np.asarray(inputs).reshape(num_classes, num_target, -1))
            batch['label'].append(np.asarray(labels).reshape(num_classes, num_target, -1))
            
        # convert to tensor
        input_tensor = torch.from_numpy(np.array(batch['input'])).permute(0,2,1,3)
        label_tensor = torch.from_numpy(np.array(batch['label'])).permute(0,2,1,3)

        image_batch = F.normalize(input_tensor, dim=-1)
        label_batch = torch.eye(num_classes)[label_tensor].squeeze(dim=-2)        
        image_batch = image_batch.reshape(batch_size, num_classes * num_target, -1) 
        label_batch = label_batch.reshape(batch_size, num_classes * num_target, -1) 
        
        Cx = image_batch[:, :num_classes*num_context, :]
        Cy = label_batch[:, :num_classes*num_context, :]
        Tx = image_batch
        Ty = label_batch
        return (Cx, Tx), (Cy, Ty)
            
        
class Heterogeneous_dataset(object):
    def __init__(self, data_source, is_train):
        self.is_train = is_train
        
        if data_source == '1D':
            pass
        if data_source == 'multidataset':
            self.generate_multidataset_folder()
        
    def generate_batch(self, batch_size, noise, is_test=False):
        num_context = 5
        num_target = 10
        
        # sine
        amp = np.random.uniform(0.1, 5.0, size=batch_size)
        phase = np.random.uniform(0.0, 2.0 * np.pi, size=batch_size)
        freq = np.random.uniform(0.8, 1.2, size=batch_size)

        # linear
        A_l = np.random.uniform(-3.0, 3.0, size=batch_size)
        b_l = np.random.uniform(-3.0, 3.0, size=batch_size)

        # quadratic
        A_q = np.random.uniform(-0.2, 0.2, size=batch_size)
        b_q = np.random.uniform(-2.0, 2.0, size=batch_size)
        c_q = np.random.uniform(-3.0, 3.0, size=batch_size)

        # cubic
        A_c = np.random.uniform(-0.1, 0.1, size=batch_size)
        b_c = np.random.uniform(-0.2, 0.2, size=batch_size)
        c_c = np.random.uniform(-2.0, 2.0, size=batch_size)
        d_c = np.random.uniform(-3.0, 3.0, size=batch_size)

        sel_set = np.zeros(batch_size)

        init_inputs = np.zeros([batch_size, 1000 if is_test else num_context+num_target, 1])
        outputs = np.zeros([batch_size, 1000 if is_test else num_context+num_target, 1])

        for b in range(batch_size):
            if is_test:
                init_inputs[b] = np.expand_dims(np.linspace(-5.0, 5.0, num=1000), axis=1)
            else:
                init_inputs[b] = np.random.uniform(-5.0, 5.0, size=(num_context+num_target, 1))
            
            sel = np.random.randint(4)
            sel_set[b] = sel
            
            if sel == 0:
                outputs[b] = amp[b] * np.sin(freq[b] * init_inputs[b]) + phase[b]
            elif sel == 1:
                outputs[b] = A_l[b] * init_inputs[b] + b_l[b]
            elif sel == 2:
                outputs[b] = A_q[b] * np.square(init_inputs[b]) + b_q[b] * init_inputs[b] + c_q[b]
            elif sel == 3:
                outputs[b] = A_c[b] * np.power(init_inputs[b], np.tile([3], init_inputs[b].shape)) + \
                             b_c[b] * np.square(init_inputs[b]) + c_c[b] * init_inputs[b] + d_c[b]
        
        init_inputs = torch.from_numpy(init_inputs)
        outputs = torch.from_numpy(outputs)
        
        rng = np.random.default_rng()
        #zero2four = np.array([10, 110, 260, 300, 480])
        random_idx = rng.permutation(init_inputs.shape[1])
        #for ele in zero2four:
        #    random_idx = np.delete(random_idx, np.where(random_idx==ele))
        #random_idx = np.concatenate([zero2four, random_idx], axis=-1)
        
        noise_shape = (batch_size, num_target, 1)
        outputs[:, random_idx[-num_target:], :] += noise * torch.randn(noise_shape)
        
        Cx = init_inputs[:, random_idx[:num_context], :]
        Cy = outputs[:, random_idx[:num_context], :]
        if self.is_train and is_test:
            Tx = init_inputs[:, random_idx[-num_target:], :] 
            Ty = outputs[:, random_idx[-num_target:], :]
        else:
            Tx = init_inputs
            Ty = outputs
        return (Cx, Tx), (Cy, Ty)#, sel_set
    
    def generate_multidataset_folder(self):
        multidataset = ['CUB_Bird', 'DTD_Texture', 'FGVC_Aircraft', 'FGVCx_Fungi']
        metatrain_folders, metavalid_folders, metatest_folders = [], [], []
        for dataset in multidataset:
            if self.is_train:
                metatrain_folders.append(
                    [os.path.join('{0}/multidataset/{1}/train'.format('../data', dataset), label) \
                     for label in os.listdir('{0}/multidataset/{1}/train'.format('../data', dataset)) \
                     if
                     os.path.isdir(os.path.join('{0}/multidataset/{1}/train'.format('../data', dataset), label)) \
                     ])
                metavalid_folders.append(
                    [os.path.join('{0}/multidataset/{1}/val'.format('../data', dataset), label) \
                     for label in os.listdir('{0}/multidataset/{1}/val'.format('../data', dataset)) \
                     if os.path.isdir(
                        os.path.join('{0}/multidataset/{1}/val'.format('../data', dataset), label)) \
                     ])
            else:
                metatest_folders.append(
                    [os.path.join('{0}/multidataset/{1}/test'.format('../data', dataset), label) \
                     for label in os.listdir('{0}/multidataset/{1}/test'.format('../data', dataset)) \
                     if os.path.isdir(
                        os.path.join('{0}/multidataset/{1}/test'.format('../data', dataset), label)) \
                     ])    
        
        self.metatrain_folders = metatrain_folders
        self.metavalid_folders = metavalid_folders
        self.metatest_folders = metatest_folders
    
    def get_images(self, paths, labels, num_context, num_target):
        sampler = lambda x: random.sample(x, num_target)
    
        tmp_list = []
        for i, path in zip(labels, paths):
            image_list = os.listdir(path)
        
            if '.ipynb_checkpoints' in image_list:
                image_list.remove('.ipynb_checkpoints')
        
            for image in sampler(image_list):
                tmp_list.append((i, os.path.join(path, image)))
    
        labels_and_filenames = []
        for idx in range(len(paths)):
            labels_and_filenames += tmp_list[idx::num_target]
        return labels_and_filenames
    
    def generate_multidataset_batch(self, batch_size, num_context, num_target, num_classes, is_test=False):
        if self.is_train:
            if is_test:
                folders = self.metavalid_folders
            else:
                folders = self.metatrain_folders
        else:
            folders = self.metatest_folders
           
        sel_set = np.zeros(batch_size)
        
        cnt = 0
        for b in range(batch_size):
            sel = np.random.randint(4)
            sel_set[b] = sel
            folder = folders[sel]
            
            sampled_classes = random.sample(folder, num_classes)
            labels_and_filenames = self.get_images(sampled_classes, range(num_classes),
                                                num_context, num_target)
            
            labels = [li[0] for li in labels_and_filenames]
            filenames = [li[1] for li in labels_and_filenames]
            
            tmp = 0
            for f in filenames:
                image = Image.open(f)
                image = np.expand_dims(image, axis=0) / 255.0
                if tmp == 0:
                    image_task = image
                    tmp += 1
                else:
                    image_task = np.concatenate((image_task, image), axis=0)
                    
            image_task = np.expand_dims(image_task, axis=0)    
            label_task = np.eye(num_classes)[labels]           
            label_task = np.expand_dims(label_task, axis=0)
            if cnt == 0:
                image_batch = image_task
                label_batch = label_task
                cnt += 1
            else:
                image_batch = np.concatenate((image_batch, image_task), axis=0)
                label_batch = np.concatenate((label_batch, label_task), axis=0)
        
        # convert to tensor
        image_batch = torch.from_numpy(image_batch).permute(0,1,4,2,3)
        label_batch = torch.from_numpy(label_batch)
        
        Cx = image_batch[:, :num_classes*num_context, :]
        Cy = image_batch[:, :num_classes*num_context, :]
        Tx = label_batch
        Ty = label_batch
        return (Cx, Tx), (Cy, Ty), sel_set
    
if __name__ == '__main__':
    from collections import deque
    import torch

    data = 'mixture'
    que_len = 100000 #100000 for train 1000 for valid 1 for test
    batch_size = 100 #100 for train and valid 1 for test
    is_train = True
    is_test = False
    
    if data == 'GP':
        loader = GaussianProcess(is_train=is_train)
    elif data == 'mixture':
        loader = Heterogeneous_dataset('1D', is_train=is_train)
        
    que = deque(maxlen=que_len)
    for idx in range(que_len):
        if idx < 0.7 * que_len:
            noise = 0.3
        elif idx < 0.9 * que_len:
            noise = 0.15
        else:
            noise = 0.0
        #noise = 0.0
        
        (Cx, Tx), (Cy, Ty) = loader.generate_batch(batch_size, noise, is_test=is_test)
        C, T = torch.cat([Cx, Cy], dim=2), torch.cat([Tx, Ty], dim=2)
        que.append((C,T))
        
    torch.save(que, f'data/{data}_{is_train}_{is_test}.pt')