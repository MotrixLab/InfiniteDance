"""The release trainer must retain the pretrained backbone it just loaded."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.dancellama import Music2DanceLlamaModel


class SmallBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 8)
        self.linear = nn.Linear(8, 8)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(0.25)

    def resize_token_embeddings(self, vocab_size):
        assert vocab_size == 20
        return self.embedding


class BackboneInitializationTest(unittest.TestCase):
    def test_pretrained_weights_survive_adapter_initialization(self):
        backbone = SmallBackbone()
        expected = {key: value.clone() for key, value in backbone.state_dict().items()}
        with patch('models.dancellama.LlamaForCausalLM.from_pretrained', return_value=backbone):
            model = Music2DanceLlamaModel(
                pretrained_model_name='fixture', require_pretrained_llama=True,
                vocab_size=20, style_embedding_dim=4, music_enc_dim=8,
                dance_enc_dim=8, llama_hidden_dim=8, bridge_hidden_dim=8,
                bridge_num_heads=2, dance_retrieval_cond_dim=6,
                music_len=4, retrieval_dance_len=8,
            )
        for key, value in model.llama.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        self.assertGreater(model.style_embedding.weight.std().item(), 0)
        torch.testing.assert_close(model.cond_projection.bias, torch.zeros_like(model.cond_projection.bias))


if __name__ == '__main__':
    unittest.main()
