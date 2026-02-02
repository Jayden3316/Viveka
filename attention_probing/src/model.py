"""
Modified transformer components for attention probing.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Optional, Tuple, Dict
import copy


class ProbeAttention(nn.Module):
    """
    Modified attention layer that can skip BOS token contributions to [PROBE].
    """
    def __init__(self, base_attention, probe_token_id: int):
        super().__init__()
        self.base_attention = base_attention
        self.probe_token_id = probe_token_id
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with BOS masking for probe token.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, 1, seq_len, seq_len] or similar
        """
        # Run base attention
        outputs = self.base_attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,  # Force attention output for masking
            **kwargs
        )
        
        attn_output = outputs[0]  # [batch_size, seq_len, hidden_dim]
        attn_weights = outputs[1]  # [batch_size, num_heads, seq_len, seq_len]
        
        # Find probe token positions (last token in sequence typically)
        # Assuming probe token is always at the last position
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Mask BOS (position 0) contributions to probe token (last position)
        # Modify attention weights: set attention from probe to BOS to 0
        if attn_weights is not None:
            # Clone to avoid in-place modification
            modified_attn_weights = attn_weights.clone()
            
            # Zero out attention from last position (probe) to first position (BOS)
            # Shape: [batch, num_heads, query_pos, key_pos]
            modified_attn_weights[:, :, -1, 0] = 0.0
            
            # Renormalize attention weights for probe token position
            probe_attn_sum = modified_attn_weights[:, :, -1, :].sum(dim=-1, keepdim=True)
            modified_attn_weights[:, :, -1, :] = modified_attn_weights[:, :, -1, :] / (probe_attn_sum + 1e-10)
            
            # Recompute attention output for probe position
            # This requires access to value vectors - alternative approach below
        
        return (attn_output,) + outputs[1:]


class ProbeAttentionV2(nn.Module):
    """
    Alternative: Modify attention mask instead of post-hoc weight manipulation.
    This is cleaner and works with any attention implementation.
    """
    def __init__(self, base_attention, probe_token_id: int, skip_bos: bool = True):
        super().__init__()
        self.base_attention = base_attention
        self.probe_token_id = probe_token_id
        self.skip_bos = skip_bos
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Modify attention mask to prevent probe from attending to BOS
        if attention_mask is not None:
            modified_mask = attention_mask.clone()
        else:
            # Create causal mask
            modified_mask = torch.ones(batch_size, 1, seq_len, seq_len, 
                                      device=hidden_states.device, dtype=torch.bool)
            modified_mask = torch.triu(modified_mask, diagonal=1)
            modified_mask = modified_mask.masked_fill(modified_mask == 1, float('-inf'))
        
        if self.skip_bos:
            # Mask BOS for probe token (assuming probe is last token)
            # Set attention score from position -1 (probe) to position 0 (BOS) to -inf
            modified_mask[:, :, -1, 0] = float('-inf')
        
        return self.base_attention(
            hidden_states,
            attention_mask=modified_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )


class UncertaintyProbe(nn.Module):
    """
    Probe head that operates on [PROBE] token representations.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 2,  # Binary: certain/uncertain
        dropout: float = 0.1,
        use_regression: bool = False
    ):
        super().__init__()
        self.use_regression = use_regression
        
        self.probe_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1 if use_regression else num_classes)
        )
        
    def forward(self, probe_representation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probe_representation: [batch_size, hidden_dim] - probe token rep
        Returns:
            logits or regression scores: [batch_size, num_classes] or [batch_size, 1]
        """
        return self.probe_head(probe_representation)


class ModelWithProbe(nn.Module):
    """
    Wrapper that combines base LLM with probe mechanisms.
    """
    def __init__(
        self,
        model_name: str,
        probe_config: Dict,
        freeze_base: bool = True
    ):
        super().__init__()
        
        self.force_regular_attention = probe_config.get('regular_attention_everywhere', False)
        
        # Load base model
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add [PROBE] token to vocabulary
        self.tokenizer = self._add_probe_token()
        self.probe_token_id = self.tokenizer.convert_tokens_to_ids('[PROBE]')
        
        # Resize model embeddings
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            
            # Keep probe token embedding trainable
            self.base_model.get_input_embeddings().weight.requires_grad = True
        
        # Replace attention layers with probe-aware versions
        self._modify_attention_layers()
        
        # Add probe head
        self.probe_classifier = UncertaintyProbe(
            hidden_dim=self.config.hidden_size,
            num_classes=probe_config.get('num_classes', 2),
            dropout=probe_config.get('dropout', 0.1),
            use_regression=probe_config.get('use_regression', False)
        )
        
        # Track attention patterns for analysis
        self.attention_patterns = []
        
    def _add_probe_token(self):
        """Add [PROBE] token to tokenizer."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model.config._name_or_path)
        
        special_tokens = {'additional_special_tokens': ['[PROBE]']}
        tokenizer.add_special_tokens(special_tokens)
        
        return tokenizer
    
    def _modify_attention_layers(self):
        """Replace attention layers with probe-aware versions."""
        # This is model-specific - example for Llama/Mistral architecture
        self.probe_attention_layers = []
        for i, layer in enumerate(self.base_model.model.layers):
            layer.self_attn = ProbeAttentionV2(
                layer.self_attn,
                self.probe_token_id,
                skip_bos=not self.force_regular_attention
            )
            self.probe_attention_layers.append(layer.self_attn)

    def _set_attention_mode(self, mode: str):
        """
        Set attention mode for all layers.
        Args:
            mode: 'regular' or 'probe'
        """
        if self.force_regular_attention:
            skip_bos = False
        else:
            skip_bos = (mode == 'probe')
            
        for layer in self.probe_attention_layers:
            layer.skip_bos = skip_bos
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_probe_logits: bool = True,
        return_generation_logits: bool = False,
        output_attentions: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with probe prediction.
        
        Returns:
            Dictionary containing:
                - probe_logits: Uncertainty predictions from probe
                - generation_logits: LM logits (optional)
                - hidden_states: All layer outputs (optional)
                - attentions: Attention weights (optional)
                - attentions: Attention weights (optional)
        """
        # Ensure correct attention mode for probing
        self._set_attention_mode('probe')

        # Run base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            **kwargs
        )
        
        # Extract probe token representation (last token, last layer)
        last_hidden_state = outputs.hidden_states[-1]  # [batch, seq_len, hidden]
        probe_repr = last_hidden_state[:, -1, :]  # [batch, hidden]
        
        # Get probe predictions
        probe_logits = self.probe_classifier(probe_repr)
        
        result = {'probe_logits': probe_logits}
        
        if return_generation_logits:
            result['generation_logits'] = outputs.logits
        
        if output_attentions:
            result['attentions'] = outputs.attentions
        
        if output_hidden_states:
            result['hidden_states'] = outputs.hidden_states
        
        return result
    
    def generate_with_probe(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate text and probe uncertainty simultaneously.
        """
        # Append [PROBE] token
        batch_size = input_ids.shape[0]
        probe_token = torch.tensor(
            [[self.probe_token_id]] * batch_size,
            device=input_ids.device
        )
        input_ids_with_probe = torch.cat([input_ids, probe_token], dim=1)
        
        # Switch to regular attention for generation
        self._set_attention_mode('regular')

        # Generate
        generated_outputs = self.base_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            **generation_kwargs
        )
        
        # Switch back to probe attention for uncertainty estimation
        self._set_attention_mode('probe')

        # Get probe prediction
        with torch.no_grad():
            probe_outputs = self.forward(
                input_ids_with_probe,
                return_probe_logits=True,
                return_generation_logits=False
            )
        
        return {
            'generated_ids': generated_outputs.sequences,
            'generation_scores': generated_outputs.scores,
            'probe_logits': probe_outputs['probe_logits']
        }


def load_model_with_probe(
    model_name: str,
    probe_config: Dict,
    checkpoint_path: Optional[str] = None
) -> ModelWithProbe:
    """Load model with probe, optionally from checkpoint."""
    model = ModelWithProbe(model_name, probe_config)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    return model