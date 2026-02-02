import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model import ModelWithProbe, ProbeAttentionV2

class MockAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args, **kwargs):
        return (torch.randn(1, 10, 32),)

class MockLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MockAttention()

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer() for _ in range(2)])

class MockBaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = MockModel()
        self.dummy_emb = nn.Embedding(100, 32)
        
    def get_input_embeddings(self):
        return self.dummy_emb
        
    def resize_token_embeddings(self, new_num):
        pass
        
    def forward(self, *args, **kwargs):
        return MagicMock(logits=torch.randn(1, 10, 100), hidden_states=[torch.randn(1, 10, 32)], attentions=[])
        
    def generate(self, *args, **kwargs):
        return MagicMock(sequences=torch.tensor([[1, 2, 3]]), scores=[])

class TestAttentionSwitching(unittest.TestCase):
    def setUp(self):
        self.mock_config = MagicMock()
        self.mock_config.hidden_size = 32
        self.mock_config._name_or_path = "mock_model"
        
        # Patch things needed for ModelWithProbe init
        self.patcher1 = patch('transformers.AutoConfig.from_pretrained', return_value=self.mock_config)
        self.patcher2 = patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=MockBaseModel(self.mock_config))
        self.patcher3 = patch('transformers.AutoTokenizer.from_pretrained')
        
        self.mock_config_loader = self.patcher1.start()
        self.mock_model_loader = self.patcher2.start()
        self.mock_tokenizer_loader = self.patcher3.start()
        
        # Setup mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.convert_tokens_to_ids.return_value = 99
        self.mock_tokenizer.__len__.return_value = 100
        self.mock_tokenizer_loader.return_value = self.mock_tokenizer

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()

    def test_default_switching(self):
        print("\nTesting default switching...")
        probe_config = {
            'num_classes': 2,
            'regular_attention_everywhere': False
        }
        
        model = ModelWithProbe("mock_model", probe_config)
        
        # Check initial state
        has_probe_attn = isinstance(model.base_model.model.layers[0].self_attn, ProbeAttentionV2)
        print(f"Layers wrapped with ProbeAttentionV2: {has_probe_attn}")
        self.assertTrue(has_probe_attn)
        
        # Verify initial mode (should be unpredictable or default, but let's check what verify_mode does)
        # Helper to check mode
        def check_mode(expected_skip_bos):
            for layer in model.probe_attention_layers:
                self.assertEqual(layer.skip_bos, expected_skip_bos)
        
        # Initialize -> force to probe mode usually in forward
        model._set_attention_mode('probe')
        check_mode(True)
        print("Set mode 'probe' -> skip_bos=True confirmed")
        
        model._set_attention_mode('regular')
        check_mode(False)
        print("Set mode 'regular' -> skip_bos=False confirmed")
        
        # Test generate_with_probe flow
        # We need to mock _set_attention_mode to verify call order, or just check the result if we could hook inputs
        # But simply checking if the logic is in place by calling generate_with_probe and seeing if it doesn't crash 
        # and checking state after. 
        # Actually generate_with_probe switches to regular, gens, then switches back to probe.
        # So at the end, it should be in probe mode.
        
        input_ids = torch.tensor([[1]])
        model.generate_with_probe(input_ids)
        check_mode(True) # Should end in probe mode
        print("generate_with_probe ended in probe mode confirmed")

    def test_force_regular(self):
        print("\nTesting forced regular attention...")
        probe_config = {
            'num_classes': 2,
            'regular_attention_everywhere': True
        }
        
        model = ModelWithProbe("mock_model", probe_config)
        
        def check_mode(expected_skip_bos):
            for layer in model.probe_attention_layers:
                self.assertEqual(layer.skip_bos, expected_skip_bos)
                
        # With forced regular, it should always be False
        model._set_attention_mode('probe')
        check_mode(False)
        print("Set mode 'probe' -> skip_bos=False (forced) confirmed")
        
        model._set_attention_mode('regular')
        check_mode(False)
        print("Set mode 'regular' -> skip_bos=False (forced) confirmed")

if __name__ == '__main__':
    unittest.main()
