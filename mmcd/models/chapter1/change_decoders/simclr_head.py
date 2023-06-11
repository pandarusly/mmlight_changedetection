import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import HEADS,LOSSES,build_loss
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
 
from timm.models.layers import create_classifier
nn.ReLU


class ContrastiveLossBASE(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
 
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

@LOSSES.register_module()
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        world_size = 1
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask
            
    def forward(self, z_i, z_j,metas=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        # if self.world_size > 1:
        #     z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

@LOSSES.register_module() 
class ContrastiveLossV2(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        world_size = 1
        self.world_size = world_size
 
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
    
    @staticmethod
    def make_positive(lst, p, batch):
        # Get all combinations of two elements from lst
        comb_lst = [(lst[i], lst[j]) for i in range(len(lst)) for j in range(i+1, len(lst))]

 
        for comb in comb_lst:
            i, j = comb
            p[i,j]=1 
            p[j,i]=1
            p[i+batch,j+batch]=1 
            p[j+batch,i+batch]=1
            p[i,j+batch]=1 
            p[j,i+batch]=1
            p[i+batch,j]=1 
            p[j+batch,i]=1 
            
        return p 


    def forward(self, z_i, z_j,metas):
        """
         
        """ 
        N = 2 * self.batch_size 
        z = torch.cat((z_i, z_j), dim=0) 
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # ----------- 制作正样本对
        positive_mask = torch.zeros((N, N), dtype=bool).to(z_i.device)
        for i in range(self.batch_size):
            positive_mask[i, self.batch_size + i] = 1
            positive_mask[self.batch_size  + i, i] = 1
 
        unique_strings, counts = np.unique(metas, return_counts=True)
        # print("unique_strings",len(unique_strings))
        # print(",metas",len(metas))
        indices=[]
        for i, s in enumerate(unique_strings): 
            indice = np.where(metas == s)[0]
            indices.append(indice)
            # print("indices",indice,len(indice))
            if len(indice)>1: 
                positive_mask = self.make_positive(indice,positive_mask,batch=self.batch_size)

        positive_samples = sim[positive_mask].reshape(N, -1)
        # ----------- 制作负样本对
        negative_mask = (~positive_mask).fill_diagonal_(False)     
        
        negative_samples = sim[negative_mask].reshape(N, -1)
 
 
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
    

@HEADS.register_module()
class SimClrHead(BaseDecodeHead):
    def __init__(self, 
                    weight=0.2,
                    loss_decode=dict(
                    type="ContrastiveLossV2",
                    batch_size=4
                    ),
                 **kwargs):
        super().__init__(
                channels=512,
                num_classes=1,
                out_channels=1,
                threshold=0.5,loss_decode=loss_decode,**kwargs)

        self.loss_decode = build_loss(loss_decode)
        # self.loss_decode = TemporalCosistenLoss(batch_size)
        self.weight = weight

        self.projection_dim = 64 

        global_pool, fc1 = create_classifier(self.in_channels,self.in_channels , pool_type='avg')

        self.embedding = global_pool

        self.projection = nn.Sequential(
            fc1,
            nn.ReLU(),
            nn.Linear(self.in_channels,self.projection_dim , bias=False),
        )


    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        f1, f2 = torch.chunk(inputs, 2, dim=1)
        
        embedding1 = self.embedding(f1)
        projection1 = self.projection(embedding1) 
        
        embedding2 = self.embedding(f2)
        projection2 = self.projection(embedding2) 

        return projection1,projection2
    

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        projection1,projection2 = self(inputs)
        # print(len(img_metas))
        get_prefix = lambda metas : np.array([meta['filename'][0].split('-')[-2]+meta['filename'][0].split('-')[-1] for meta in  metas])
        # print(len(img_metas),get_prefix(img_metas))
        
        loss = dict() 
        losses = self.loss_decode(projection1,projection2,get_prefix(img_metas))*self.weight
        loss['loss_simclr'] = losses

        return loss

    def forward_test(self, inputs, img_metas, test_cfg):
        return super().forward_test(inputs, img_metas, test_cfg)

@HEADS.register_module()
class SimClrHeadForPublic(SimClrHead):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        projection1,projection2 = self(inputs)
        # print(len(img_metas))
        get_prefix = lambda metas : np.array([meta['filename'][0] for meta in  metas])
        # print(len(img_metas),get_prefix(img_metas))
        
        loss = dict() 
        losses = self.loss_decode(projection1,projection2,get_prefix(img_metas))*self.weight
        loss['loss_simclr'] = losses

        return loss

# from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot