"""Contains class definition of some baseline metric learning models."""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer_hybrid import HybridEmbed    
from metric_learning.loss import ArcFaceLoss, ArcFaceLossAdaptiveMargin

class ArcMarginProduct(nn.Module):
    """ArcMarginProduct operation.

    Calculates the cosine of the angle between the embeddings and their
    corresponding centers (represented by the weight matrix).
    
    Attributes:
        weight: initialized weight matrix to map embeddings to output classes
        k: Number of subcenter for each class
        out_features: number of output classes
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitializes the weight matrix using xavier_uniform_"""
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        """Perform cosine calculation
        
        Args:
            features: embedding vectors of current minibatch

        Returns:
            A vector of cosine between embeddings and their corresponding centers
            (represented by the weight matrix)
        """
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine

class ArcMarginProductSubcenter(nn.Module):
    """ArcMarginProduct operation with subcenter configuration.

    Calculates the cosine of the angle between the embeddings and their
    corresponding centers (represented by the weight matrix), also use subcenters.
    
    Attributes:
        weight: initialized weight matrix to map embeddings to output classes
        k: Number of subcenter for each class
        out_features: number of output classes
    """
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features
        
    def reset_parameters(self):
        """Reinitialize the weight matrix using uniform_."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, features):
        """Performs cosine calculation.
        
        Args:
            features: embedding vectors of current minibatch

        Returns:
            A vector of cosine between embeddings and their corresponding centers
            (represented by the weight matrix)
        """
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine

def gem(features, p=3, eps=1e-6):
    """Performs GeM pooling.
        
    Args:
        features: A matrix of shape (n, num_channels, H, W), a feature map outputed by a convolutional layer
        p: the exponential number
        eps: small number to prevent the base of A^x not to equal 0

    Returns:
        A vector of shape (n, num_channels) resulted from the pooling action to squeeze the W and H dimension
    """
    return F.avg_pool2d(features.clamp(min=eps).pow(p), (features.size(-2), features.size(-1))).pow(1./p)

class GeM(nn.Module):
    """Performs GeM pooling as in the paper: https://arxiv.org/pdf/1711.02512.pdf.
    
        Attributes:
            p: the exponential number
            eps: small number to prevent the base of A^x not to equal 0
    """
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        """Performs GeM pooling."""
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class MultiAtrousModule(nn.Module):
    """MultiAtrousModule implemented following: https://arxiv.org/pdf/2108.02927.pdf.
        
        Apply 3 different dilational convolution to extract features

        Attributes:
            d0: the 1st dilational convolution
            d1: the 2nd dilational convolution
            d2: the 3rd dilational convolution
            relu: ReLU activation function
        """
    def __init__(self, in_chans, out_chans, dilations):
        super(MultiAtrousModule, self).__init__()
        
        self.d0 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[0],padding='same')
        self.d1 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[1],padding='same')
        self.d2 = nn.Conv2d(in_chans, 512, kernel_size=3, dilation=dilations[2],padding='same')
        self.conv1 = nn.Conv2d(512 * 3, out_chans, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        """Performs the convolution operations and concatenates the outputs."""
        x0 = self.d0(x)
        x1 = self.d1(x)
        x2 = self.d2(x)
        x = torch.cat([x0,x1,x2],dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        return x

class SpatialAttention2d(nn.Module):
    """Spatial attention operration on the input feature map.
    
    Attributes:
        conv1, conv2: convolution layers
        bn1: batch norm layer
        act1: relu activation
        softplus: an activation layer (smoothen version of ReLU)
    """
    def __init__(self, in_c):
        """
            Args:
                in_c: input channel of the feature map
        """
        super(SpatialAttention2d, self).__init__()
        self.conv1 = nn.Conv2d(in_c, 1024, 1, 1)
        self.bn = nn.BatchNorm2d(1024)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(1024, 1, 1, 1)
        self.softplus = nn.Softplus(beta=1, threshold=20) # use default setting.

    def forward(self, x):
        """Performs the attention map multiplication.
            
        Args:
            x : spatial feature map. (b x c x w x h)

        Return:
            1, The feature map after applying attention multiplication
            2, The attention map
        """
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act1(x)
        x = self.conv2(x)

        att_score = self.softplus(x)
        feature_map_norm = F.normalize(x, p=2, dim=1)
        att = att_score.expand_as(feature_map_norm)
        
        x = att * feature_map_norm
        return x, att_score

class OrthogonalFusion(nn.Module):
    """A module that combine local features and global features using orthogonal fusion.
    
    Detail can be found in: https://arxiv.org/pdf/2108.02927.pdf
    """
    def __init__(self):
        super(OrthogonalFusion, self).__init__()

    def forward(self, fl, fg):
        """Fuse local features and global features to a unified feature map."""
        bs, c, w, h = fl.shape
        
        fl_dot_fg = torch.bmm(fg[:,None,:],fl.reshape(bs,c,-1))
        fl_dot_fg = fl_dot_fg.reshape(bs,1,w,h)
        fg_norm = torch.norm(fg, dim=1)
        
        fl_proj = (fl_dot_fg / fg_norm[:,None,None,None]) * fg[:,:,None,None]
        fl_orth = fl - fl_proj
        
        f_fused = torch.cat([fl_orth,fg[:,:,None,None].repeat(1,1,w,h)],dim=1)
        return f_fused  

class SimpleArcFaceModel(nn.Module):
    """A CNN backbone classifier with default ArcFace loss function.
    
    The module can have various timm CNN backbone with names complying with those of timm.
    ArcFace is included as loss function. It can either be marignal adaptive or manually configured.
    Also, ArcFace can have subcenters
    
    Attributes:
        backbone: a torch module that is created using timm api: create_model()
        n_classes: number of classes
        embedding_size: size of the embedding vector
        global_pool: pooling method ('gem' or 'avg')
        neck: a sequence of layers mapping pooled features to embeddings
        head: 
            ArcMarginProduct, which takes embeddings as input and calculates the logits
            (the cosine between the embeddings and their centers)
        loss_fn: ArcFace loss function with either adaptive margin or not
        device: a device to create the model on
    """
    def __init__(self, backbone_name, backbone_pretrained=None, 
                n_classes=10000, embedding_size=512, global_pool='gem', margin=0.5, scale=64,
                sub_center=False, adaptive_margin=False, arcface_m_x = 0.45,
                arcface_m_y = 0.05, label_frequency=None, device='cuda:0'):
        """
        Args:
            backbone_name: timm CNN backbone name
            backbone_pretrained: 
                either boolean to allow pretrained weights to be downloaded,
                or string denoting the path to the saved weights in the local machine.
            n_classes: number of classes
            embedding_size: size of the embedding vector
            global_pool: pooling method ('gem' or 'avg')
            margin: ArcFace margin parameter (only matters if adaptive_margin=False)
            scale: ArcFace scale parameter
            sub_center: use subcenter for ArcFace instead of single center
            adaptive_margin: whether to allow a learnable margin for ArcFace
            arcface_m_x: # TOCOMMENT
            arcface_m_y: # TOCOMMENT
            label_frequency: frequency of every class in training set
            device: a device to create the model on
        """
        super(SimpleArcFaceModel, self).__init__()
        self.n_classes = n_classes
        
        if backbone_pretrained is not None:
            if type(backbone_pretrained) == bool:
                self.backbone = timm.create_model(backbone_name, pretrained=backbone_pretrained,
                                                features_only=True, num_classes=0,
                                                global_pool='')
            elif type(backbone_pretrained) == str:
                self.backbone = timm.create_model(backbone_name, pretrained=False,
                                                features_only=True, num_classes=0,
                                                global_pool='')
        
                self.backbone.load_state_dict(torch.load(backbone_pretrained))
                print('Loaded pretrained model:', backbone_pretrained)

        if global_pool == 'gem':
            self.global_pool = GeM(p_trainable=True)
        elif global_pool == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_size = embedding_size

        self.neck = nn.Sequential(
                nn.Linear(self.backbone.feature_info[-1]['num_chs'], self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )

        if sub_center:
            self.head = ArcMarginProductSubcenter(self.embedding_size, self.n_classes)
        else:
            self.head = ArcMarginProduct(self.embedding_size, self.n_classes)

        if adaptive_margin:
            if label_frequency is None:
                raise ValueError('when adaptive_margin is True, please parse label_frequency of the dataset')
            tmp = np.sqrt(1 / np.sqrt(label_frequency.sort_index().values))
            init_margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * arcface_m_x + arcface_m_y
            self.loss_fn = ArcFaceLossAdaptiveMargin(margins=init_margins,
                                                        n_classes=n_classes, s=scale,
                                                        device=device)
        else:
            self.loss_fn = ArcFaceLoss(scale, margin, device=device)

        # to device
        self.device = device
        self.to(self.device)
        
    def forward(self, batch):
        """
        Args:
            batch:
                input minibatch in the form of a dictionary
                contains either 2 keys input and target (is_training) or key input only (not is_training)
                input shape: Nx3xHxW
                target shape: Nxn_classes

        Returns:
            A dictionary contain those keys:
                loss: loss between model's prediction and the target (equals to 0 if not is_training)
                target: target of classification (None if not is_training)
                preds_conf: confidence scores of the prediction
                preds_cls: predicted labels of th prediction
                embeddings: embeddings vectors
        """
        x = batch['input']
        batch_size = x.shape[0]

        features = self.backbone(x)
        if type(features) == list:
            features = features[-1]

        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.view(batch_size,-1)

        embeddings = self.neck(pooled_features)
        logits = self.head(embeddings)

        preds = logits.softmax(1)
        
        preds_conf, preds_cls = preds.max(1)

        if self.training:
            loss = self.loss_fn(logits, batch['target'].long())
            target = batch['target']
        else:
            target = None
            loss = torch.zeros((1),device=self.device)
        return {'loss': loss, "target": target, "preds_conf":preds_conf,'preds_cls':preds_cls,
                'embeddings': embeddings
                }

class SwinTrArcFaceModel(nn.Module):
    """A Swin Transformer backbone classifier with default ArcFace loss function.
    
    The module can have various timm Swin Transformer backbone with names complying with those of timm.
    ArcFace is included as loss function. It can either be marignal adaptive or manually configured.
    Also, ArcFace can have subcenters
    
    Attributes:
        backbone: a torch module that is created using timm api: create_model()
        n_classes: number of classes
        embedding_size: size of the embedding vector
        global_pool: pooling method ('gem' or 'avg')
        neck: a sequence of layers mapping pooled features to embeddings
        head: 
            ArcMarginProduct, which takes embeddings as input and calculates the logits
            (the cosine between the embeddings and their centers)
        loss_fn: ArcFace loss function with either adaptive margin or not
        device: a device to create the model on
    """
    def __init__(self, backbone_name, backbone_pretrained=None, 
                n_classes=10000, embedding_size=512, margin=0.5, scale=64,
                sub_center=False, adaptive_margin=False, arcface_m_x = 0.45,
                arcface_m_y = 0.05, label_frequency=None, device='cuda:0'):
        """
        Args:
            backbone_name: timm Swin Transformer backbone name
            backbone_pretrained: 
                either boolean to allow pretrained weights to be downloaded,
                or string denoting the path to the saved weights in the local machine.
            n_classes: number of classes
            embedding_size: size of the embedding vector
            global_pool: pooling method ('gem' or 'avg')
            margin: ArcFace margin parameter (only matters if adaptive_margin=False)
            scale: ArcFace scale parameter
            sub_center: use subcenter for ArcFace instead of single center
            adaptive_margin: whether to allow a learnable margin for ArcFace
            arcface_m_x: # TOCOMMENT
            arcface_m_y: # TOCOMMENT
            label_frequency: frequency of every class in training set
            device: a device to create the model on
        """
        super(SwinTrArcFaceModel, self).__init__()
        self.n_classes = n_classes
        
        if backbone_pretrained is not None:
            if type(backbone_pretrained) == bool:
                self.backbone = timm.create_model(backbone_name, pretrained=backbone_pretrained)
            elif type(backbone_pretrained) == str:
                self.backbone = timm.create_model(backbone_name, pretrained=False)
        
                self.backbone.load_state_dict(torch.load(backbone_pretrained))
                print('Loaded pretrained model:', backbone_pretrained)

        self.embedding_size = embedding_size

        self.neck = nn.Sequential(
                nn.Linear(self.backbone.head.in_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        self.backbone.head = nn.Identity()

        if sub_center:
            self.head = ArcMarginProductSubcenter(self.embedding_size, self.n_classes)
        else:
            self.head = ArcMarginProduct(self.embedding_size, self.n_classes)

        if adaptive_margin:
            if label_frequency is None:
                raise ValueError('when adaptive_margin is True, please parse label_frequency of the dataset')
            tmp = np.sqrt(1 / np.sqrt(label_frequency.sort_index().values))
            init_margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * arcface_m_x + arcface_m_y
            self.loss_fn = ArcFaceLossAdaptiveMargin(margins=init_margins,
                                                        n_classes=n_classes, s=scale,
                                                        device=device)
        else:
            self.loss_fn = ArcFaceLoss(scale, margin, device=device)

        # to device
        self.device = device
        self.to(self.device)
        
    def forward(self, batch):
        """
        Args:
            batch:
                input minibatch in the form of a dictionary
                contains either 2 keys input and target (is_training) or key input only (not is_training)
                input shape: Nx3xHxW
                target shape: Nxn_classes

        Returns:
            A dictionary contain those keys:
                loss: loss between model's prediction and the target (equals to 0 if not is_training)
                target: target of classification (None if not is_training)
                preds_conf: confidence scores of the prediction
                preds_cls: predicted labels of th prediction
                embeddings: embeddings vectors
        """
        x = batch['input']
        batch_size = x.shape[0]

        pooled_features = self.backbone(x)
        pooled_features = pooled_features.view(batch_size,-1)

        embeddings = self.neck(pooled_features)
        logits = self.head(embeddings)

        preds = logits.softmax(1)
        
        preds_conf, preds_cls = preds.max(1)

        if self.training:
            loss = self.loss_fn(logits, batch['target'].long())
            target = batch['target']
        else:
            target = None
            loss = torch.zeros((1),device=self.device)
        return {'loss': loss, "target": target, "preds_conf":preds_conf,'preds_cls':preds_cls,
                'embeddings': embeddings
                }


class DOLGArcFaceModel(SimpleArcFaceModel):
    """A CNN backbone classifier with Othogonal Fusion.
    
    The module can have various timm CNN backbone with names complying with those of timm.
    ArcFace is included as loss function. It can either be marignal adaptive or manually configured.
    Also, ArcFace can have subcenters and margins can be adaptive corresponding to class frequency
    
    Attributes:
        backbone: a torch module that is created using timm api: create_model()
        n_classes: number of classes
        embedding_size: size of the embedding vector
        fusion_pool: pooling method applied after fusing the local features and the global ones
        neck: a sequence of layers mapping pooled features to embeddings
        mam: a MultiAtrousModule which performs 3 different dilation convolution operations (local branch)
        conv_g, bn_g, act_g: a series of operations (convolution, batchnorm, activation) on global features (global branch)
        attention2d: SpatialAttention module that extracts local features (local branch)
        fusion: OthogonalFusion module that fuses the local and global features
        head: 
            ArcMarginProduct, which takes embeddings as input and calculates the logits
            (the cosine between the embeddings and their centers)
        loss_fn: ArcFace loss function with either adaptive margin or not
        device: a device to create the model on
    """
    def __init__(self, backbone_name, backbone_pretrained=None, 
                n_classes=10000, embedding_size=512, global_pool='gem', margin=0.5, scale=64,
                sub_center=False, adaptive_margin=False, arcface_m_x = 0.45,
                arcface_m_y = 0.05, label_frequency=None,
                dilations=[6,12,18], device='cuda:0'):
        """
        Args:
            backbone_name: timm CNN backbone name
            backbone_pretrained: 
                either boolean to allow pretrained weights to be downloaded,
                or string denoting the path to the saved weights in the local machine.
            n_classes: number of classes
            embedding_size: size of the embedding vector
            global_pool: pooling method ('gem' or 'avg')
            margin: ArcFace margin parameter (only matters if adaptive_margin=False)
            scale: ArcFace scale parameter
            sub_center: use subcenter for ArcFace instead of single center
            adaptive_margin: whether to allow a learnable margin for ArcFace
            arcface_m_x: # TOCOMMENT
            arcface_m_y: # TOCOMMENT
            label_frequency: frequency of every class in training set
            dilations: a list with 3 elements denoting the dilation sizes of the 3 convolutions in MultiAtrousModule
            device: a device to create the model on
        """
        super(DOLGArcFaceModel, self).__init__(backbone_name, backbone_pretrained,
                                                n_classes, embedding_size, global_pool, margin, scale, 
                                                sub_center, adaptive_margin, arcface_m_x,
                                                arcface_m_y, label_frequency, device)
               
        self.n_classes = n_classes
        if backbone_pretrained is not None:
            if type(backbone_pretrained) == bool:
                self.backbone = timm.create_model(backbone_name, pretrained=backbone_pretrained,
                                                features_only=True, num_classes=0,
                                                global_pool='')
            elif type(backbone_pretrained) == str:
                self.backbone = timm.create_model(backbone_name, pretrained=False,
                                                features_only=True, num_classes=0,
                                                global_pool='')
        
                self.backbone.load_state_dict(torch.load(backbone_pretrained))
                print('Loaded pretrained model:', backbone_pretrained)
        
        backbone_out = self.backbone.feature_info[-1]['num_chs']
        backbone_out_1 = self.backbone.feature_info[-2]['num_chs']

        feature_dim_l_g = 1024
        fusion_out = 2 * feature_dim_l_g

        self.fusion_pool = nn.AdaptiveAvgPool2d(1)

        self.neck = nn.Sequential(
                nn.Linear(fusion_out, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        
        self.mam = MultiAtrousModule(backbone_out_1, feature_dim_l_g, dilations)
        self.conv_g = nn.Conv2d(backbone_out,feature_dim_l_g,kernel_size=1)
        self.bn_g = nn.BatchNorm2d(feature_dim_l_g, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.act_g =  nn.SiLU(inplace=True)
        self.attention2d = SpatialAttention2d(feature_dim_l_g)
        self.fusion = OrthogonalFusion()

        # to device
        self.device = device
        self.to(self.device)
        
    def forward(self, batch):
        """
        Args:
            batch:
                input minibatch in the form of a dictionary
                contains either 2 keys input and target (is_training) or key input only (not is_training)
                input shape: Nx3xHxW
                target shape: Nxn_classes

        Returns:
            A dictionary contain those keys:
                loss: loss between model's prediction and the target (equals to 0 if not is_training)
                target: target of classification (None if not is_training)
                preds_conf: confidence scores of the prediction
                preds_cls: predicted labels of th prediction
                embeddings: embeddings vectors
        """
        x = batch['input']

        x = self.backbone(x)
        
        x_l = x[-2]
        x_g = x[-1]
        
        x_l = self.mam(x_l)
        x_l, att_score = self.attention2d(x_l)
        
        x_g = self.conv_g(x_g)
        x_g = self.bn_g(x_g)
        x_g = self.act_g(x_g)
        
        x_g = self.global_pool(x_g)
        x_g = x_g[:,:,0,0]
        
        x_fused = self.fusion(x_l, x_g)
        x_fused = self.fusion_pool(x_fused)
        x_fused = x_fused[:,:,0,0]        
        
        x_emb = self.neck(x_fused)

        logits = self.head(x_emb)
        preds = logits.softmax(1)
        
        preds_conf, preds_cls = preds.max(1)

        if self.training:
            loss = self.loss_fn(logits, batch['target'].long())
            target = batch['target']
        else:
            target = None
            loss = torch.zeros((1),device=self.device)
        return {'loss': loss, "target": target, "preds_conf":preds_conf,'preds_cls':preds_cls,
                'embeddings': x_emb
                }
    
class HybridSwinTransformer(nn.Module):
    """Still in fixing, hence no comments"""
    def __init__(self, backbone_name, embedder_name, backbone_pretrained=None, 
                embedder_pretrained=None, embedder_blocks=[1], image_size=768,
                n_classes=10000, embedding_size=512, global_pool='gem', margin=0.5, scale=64,
                sub_center=False, adaptive_margin=False, arcface_m_x = 0.45,
                arcface_m_y = 0.05, label_frequency=None, freeze_backbone_head=False, device='cuda:0'):
        
        super(HybridSwinTransformer, self).__init__()

        self.n_classes = n_classes
        if backbone_pretrained is not None:
            if type(backbone_pretrained) == bool:
                self.backbone = timm.create_model(backbone_name, pretrained=backbone_pretrained,
                                                num_classes=0,
                                                global_pool='')
            elif type(backbone_pretrained) == str:
                self.backbone = timm.create_model(backbone_name, pretrained=False,
                                                 num_classes=0,
                                                global_pool='')
                self.backbone.load_state_dict(torch.load(backbone_pretrained))
                print('Loaded pretrained backbone:', backbone_pretrained)

        if embedder_pretrained is not None:
            if type(embedder_pretrained) == bool:
                embedder = timm.create_model(embedder_name,
                                        pretrained=embedder_pretrained, 
                                        in_chans=3,
                                        features_only=True, out_indices=embedder_blocks)
            if type(embedder_pretrained) == str:
                embedder = timm.create_model(embedder_name, 
                                        pretrained=False, 
                                        in_chans=3,
                                        features_only=True, out_indices=embedder_blocks)
                embedder.load_state_dict(torch.load(embedder_pretrained))
                print('Loaded pretrained embedder:', embedder_pretrained)
        
        print('Embedder output blocks:', embedder_blocks)
        self.backbone.patch_embed = HybridEmbed(embedder,img_size=image_size, 
                                              patch_size=1, 
                                              feature_size=self.backbone.patch_embed.grid_size, 
                                              in_chans=3, 
                                              embed_dim=self.backbone.embed_dim)
 
        if global_pool == 'gem':
            self.global_pool = GeM(p_trainable=True)
        elif global_pool == 'avg':
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_size = embedding_size

        self.neck = nn.Sequential(
                nn.Linear(self.backbone.num_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )

        if sub_center:
            self.head = ArcMarginProductSubcenter(self.embedding_size, self.n_classes)
        else:
            self.head = ArcMarginProduct(self.embedding_size, self.n_classes)

        if adaptive_margin:
            if label_frequency is None:
                raise ValueError('when adaptive_margin is True, please parse label_frequency of the dataset')
            tmp = np.sqrt(1 / np.sqrt(label_frequency.sort_index().values))
            init_margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * arcface_m_x + arcface_m_y
            self.loss_fn = ArcFaceLossAdaptiveMargin(margins=init_margins,
                                                        n_classes=n_classes, s=scale,
                                                        device=device)
        else:
            self.loss_fn = ArcFaceLoss(scale, margin, device=device)

        if freeze_backbone_head:
            for name, param in self.named_parameters():
                if not 'patch_embed' in name:
                    param.requires_grad = False

        # to device
        self.device = device
        self.to(self.device)


    def forward(self, batch):

        x = batch['input']

        x = self.backbone(x)

        x_emb = self.neck(x)

        logits = self.head(x_emb)
        preds = logits.softmax(1)
        
        preds_conf, preds_cls = preds.max(1)

        if self.training:
            loss = self.loss_fn(logits, batch['target'].long())
            target = batch['target']
        else:
            target = None
            loss = torch.zeros((1),device=self.device)
        return {'loss': loss, "target": target, "preds_conf":preds_conf,'preds_cls':preds_cls,
                'embeddings': x_emb
                }