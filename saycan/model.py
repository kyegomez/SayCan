from torch.nn import Module
from transformers import AutoTokenizer

from saycan.transformer import (
    Decoder,
    Transformer,
    ViTransformerWrapper,
    Encoder
)


from saycan.autoregressive import AutoregressiveWrapper

class SayCanTokenizer:
    def __init__(self):
        self.tokenizer= AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b",
            eos_token="<eos>",
            pad_token="<pad>",
            extra_ids=0,
            model_max_length=8192
        )

    def tokenize_texts(self, texts):
        return self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).input_ids
    
    def decode(self, texts):
        return self.tokenizer.decode(texts)
    
    def __len__(self):
        num_tokens = len(self.tokenizer)
        return num_tokens



class SayCan(Module):
    """
    SayCan is a transformer-based model architecture. It initializes with 
    a Transformer and AutoregressiveWrapper with default or user-specified parameters.
    """
    def __init__(self, 
                 num_tokens=50432, 
                 max_seq_len=8192, 
                 dim=2560, 
                 depth=32, 
                 dim_head=128, 
                 heads=24,
                 use_abs_pos_emb=False, 
                 alibi_pos_bias=True, 
                 alibi_num_heads=12, 
                 rotary_xpos=True,
                 attn_flash=True, 
                 attn_kv_heads = 2,
                 qk_norm=True, 
                 attn_qk_norm=True, 
                 attn_qk_norm_dim_scale=True, 
                 ):
        """
        Initialize the model with specified or default parameters.
        Args:
        - num_tokens: Number of tokens in the vocabulary
        - max_seq_len: Maximum sequence length
        - dim: Dimension of the model
        - depth: Depth of the model
        - dim_head: Dimension of the model head
        - heads: Number of heads
        - use_abs_pos_emb: Whether to use absolute position embedding
        - alibi_pos_bias: Alibi position bias
        - alibi_num_heads: Number of alibi heads
        - rotary_xpos: Rotary position
        - attn_flash: Attention flash
        - deepnorm: Deep normalization
        - shift_tokens: Number of tokens to shift
        - attn_one_kv_head: Attention one key/value head
        - qk_norm: Query-key normalization
        - attn_qk_norm: Attention query-key normalization
        - attn_qk_norm_dim_scale: Attention query-key normalization dimension scale
        - embedding_provider: Embedding provider module
        """
        super().__init__()

        try:
            self.SayCan = Transformer(
                num_tokens=num_tokens,
                max_seq_len=max_seq_len,
                use_abs_pos_emb=use_abs_pos_emb,
                attn_layers=Decoder(
                    dim=dim,
                    depth=depth,
                    dim_head=dim_head,
                    heads=heads,
                    alibi_pos_bias=alibi_pos_bias,
                    alibi_num_heads=alibi_num_heads,
                    rotary_xpos=rotary_xpos,
                    attn_flash=attn_flash,
                    attn_kv_heads=attn_kv_heads,
                    qk_norm=qk_norm,
                    attn_qk_norm=attn_qk_norm,
                    attn_qk_norm_dim_scale=attn_qk_norm_dim_scale
                )
            )

            self.decoder = AutoregressiveWrapper(self.SayCan)

        except Exception as e:
            print("Failed to initialize SayCan: ", e)
            raise

    def forward(self, text_tokens, **kwargs):
        """
        Forward pass through the model. It expects the input text_tokens.
        Args:
        - text_tokens: Input tokens
        - kwargs: Other arguments
        Returns:
        - output from the decoder
        """
        try:
            model_input = self.decoder.forward(text_tokens)[0]
            return self.decoder(model_input, padded_x=model_input[0])
        except Exception as e:
            print("Failed in forward method: ", e)
            raise


class SayCanMultiModal(Module):
    def __init__(
            self, 
            image_size=256, 
            patch_size=32, 
            encoder_dim=512, 
            encoder_depth=6, 
            encoder_heads=8,
            num_tokens=20000, 
            max_seq_len=1024, 
            decoder_dim=512, 
            decoder_depth=6, 
            decoder_heads=8, 
            alibi_num_heads=4,
            use_abs_pos_emb=False,
            cross_attend=True,
            alibi_pos_bias=True,
            rotary_xpos=True,
            attn_flash=True,
            qk_norm=True
        ):
        super(SayCanMultiModal, self).__init__()
        
        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim,
                depth=encoder_depth,
                heads=encoder_heads
            )
        )

        self.decoder = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                cross_attend=cross_attend,
                alibi_pos_bias=alibi_pos_bias,
                alibi_num_heads=alibi_num_heads,
                rotary_xpos=rotary_xpos,
                attn_flash=attn_flash,
                qk_norm=qk_norm,
            )
        )

    def forward(self, img, text):
        try:    
            encoded = self.encoder(img, return_embeddings=True)
            return self.decoder(text, context=encoded)
        except Exception as error:
            print(f"Failed in forward method: {error}")
            raise