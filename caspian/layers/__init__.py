from .Layer import Layer
from .Linear import Linear
from .Dense import Dense
from .Reshape import Reshape
from .BatchNorm import BatchNorm
from .LayerNorm import LayerNorm
from .GroupNorm import GroupNorm
from .InstanceNorm import InstanceNorm
from .RMSNorm import RMSNorm
from .Conv1D import Conv1D
from .Conv1DTranspose import Conv1DTranspose
from .Upsampling1D import Upsampling1D
from .Pooling1D import Pooling1D
from .Conv2D import Conv2D
from .Conv2DTranspose import Conv2DTranspose
from .Pooling2D import Pooling2D
from .Upsampling2D import Upsampling2D
from .Conv3D import Conv3D
from .Conv3DTranspose import Conv3DTranspose
from .Pooling3D import Pooling3D
from .Upsampling3D import Upsampling3D
from .ConvND import ConvND
from .Bilinear import Bilinear
from .Container import Container
from .Embedding import Embedding
from .Dropout import Dropout
from .Sequence import Sequence
from .Add import Add
from .Mult import Mult
from .MatMul import MatMul
from .Concat import Concat
from .PositionalEncoding import PositionalEncoding
from .Attention import Attention
from .MultiHeadAttention import MultiHeadAttention
from .CosineSimilarity import CosineSimilarity

layer_dict: dict[str, Layer] = {"Layer":Layer,
                                "Linear":Linear,
                                "Dense":Dense,
                                "Reshape":Reshape,
                                "BatchNorm":BatchNorm,
                                "LayerNorm":LayerNorm,
                                "GroupNorm":GroupNorm,
                                "InstanceNorm":InstanceNorm,
                                "RMSNorm":RMSNorm,
                                "Conv1D":Conv1D,
                                "Conv1DTranspose":Conv1DTranspose,
                                "Conv2D":Conv2D,
                                "Conv2DTranspose":Conv2DTranspose,
                                "Conv3D":Conv3D,
                                "Conv3DTranspose":Conv3DTranspose,
                                "ConvND":ConvND,
                                "Upsampling1D":Upsampling1D,
                                "Upsampling2D":Upsampling2D,
                                "Upsampling3D":Upsampling3D,
                                "Pooling1D":Pooling1D,
                                "Pooling2D":Pooling2D,
                                "Pooling3D":Pooling3D,
                                "Bilinear":Bilinear,
                                "Container":Container,
                                "Embedding":Embedding,
                                "Dropout":Dropout,
                                "Sequence":Sequence,
                                "Add":Add,
                                "Mult":Mult,
                                "MatMul":MatMul,
                                "Concat":Concat,
                                "PositionalEncoding":PositionalEncoding,
                                "Attention":Attention,
                                "MultiHeadAttention":MultiHeadAttention,
                                "CosineSimilarity":CosineSimilarity
                                }