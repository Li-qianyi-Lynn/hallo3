# pylint: disable=R0901
# src/models/wav2vec.py

"""
This module defines the Wav2Vec model, which is a pre-trained model for speech recognition and understanding.
It inherits from the Wav2Vec2Model class in the transformers library and provides additional functionalities
such as feature extraction and encoding.

Classes:
    Wav2VecModel: Inherits from Wav2Vec2Model and adds additional methods for feature extraction and encoding.

Functions:
    linear_interpolation: Interpolates the features based on the sequence length.
"""

import torch.nn.functional as F
from transformers import Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput


class Wav2VecModel(Wav2Vec2Model):
    """
    Wav2VecModel is a custom model class that extends the Wav2Vec2Model class from the transformers library. 
    It inherits all the functionality of the Wav2Vec2Model and adds additional methods for feature extraction and encoding.
    ...

    Attributes:
        base_model (Wav2Vec2Model): The base Wav2Vec2Model object.

    Methods:
        forward(input_values, seq_len, attention_mask=None, mask_time_indices=None
        , output_attentions=None, output_hidden_states=None, return_dict=None):
            Forward pass of the Wav2VecModel. 
            It takes input_values, seq_len, and other optional parameters as input and returns the output of the base model.

        feature_extract(input_values, seq_len):
            Extracts features from the input_values using the base model.

        encode(extract_features, attention_mask=None, mask_time_indices=None, output_attentions=None, output_hidden_states=None, return_dict=None):
            Encodes the extracted features using the base model and returns the encoded features.
    """
    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the Wav2Vec model.

        Args:
            self: The instance of the model.
            input_values: The input values (waveform) to the model.
            seq_len: The sequence length of the input values.
            attention_mask: Attention mask to be used for the model.
            mask_time_indices: Mask indices to be used for the model.
            output_attentions: If set to True, returns attentions.
            output_hidden_states: If set to True, returns hidden states.
            return_dict: If set to True, returns a BaseModelOutput instead of a tuple.

        Returns:
            The output of the Wav2Vec model.
        """
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        print(f"[AUDIO_DEBUG] Wav2VecModel.forward(): 输入 input_values shape={input_values.shape}, seq_len={seq_len}")

        extract_features = self.feature_extractor(input_values)
        print(f"[AUDIO_DEBUG]   feature_extractor 后: shape={extract_features.shape}")
        extract_features = extract_features.transpose(1, 2)
        print(f"[AUDIO_DEBUG]   transpose(1,2) 后: shape={extract_features.shape}")
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)
        print(f"[AUDIO_DEBUG]   linear_interpolation 后: shape={extract_features.shape}")

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        print(f"[AUDIO_DEBUG]   feature_projection 后: hidden_states shape={hidden_states.shape}")
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]
        print(f"[AUDIO_DEBUG]   encoder 后: last_hidden_state shape={hidden_states.shape}")
        if hasattr(encoder_outputs, 'hidden_states') and encoder_outputs.hidden_states is not None:
            print(f"[AUDIO_DEBUG]   encoder 后: num_hidden_states={len(encoder_outputs.hidden_states)}, 每层 shape={encoder_outputs.hidden_states[0].shape}")

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        """
        Extracts features from the input values and returns the extracted features.

        Parameters:
        input_values (torch.Tensor): The input values to be processed.
        seq_len (torch.Tensor): The sequence lengths of the input values.

        Returns:
        extracted_features (torch.Tensor): The extracted features from the input values.
        """
        print(f"[AUDIO_DEBUG] Wav2VecModel.feature_extract(): 输入 input_values shape={input_values.shape}, seq_len={seq_len}")
        extract_features = self.feature_extractor(input_values)
        print(f"[AUDIO_DEBUG]   feature_extractor 后: shape={extract_features.shape}")
        extract_features = extract_features.transpose(1, 2)
        print(f"[AUDIO_DEBUG]   transpose(1,2) 后: shape={extract_features.shape}")
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)
        print(f"[AUDIO_DEBUG]   linear_interpolation 后: shape={extract_features.shape}")

        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Encodes the input features into the output space.

        Args:
            extract_features (torch.Tensor): The extracted features from the audio signal.
            attention_mask (torch.Tensor, optional): Attention mask to be used for padding.
            mask_time_indices (torch.Tensor, optional): Masked indices for the time dimension.
            output_attentions (bool, optional): If set to True, returns the attention weights.
            output_hidden_states (bool, optional): If set to True, returns all hidden states.
            return_dict (bool, optional): If set to True, returns a BaseModelOutput instead of the tuple.

        Returns:
            The encoded output features.
        """
        self.config.output_attentions = True

        print(f"[AUDIO_DEBUG] Wav2VecModel.encode(): 输入 extract_features shape={extract_features.shape}")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        print(f"[AUDIO_DEBUG]   feature_projection 后: hidden_states shape={hidden_states.shape}")
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]
        print(f"[AUDIO_DEBUG]   encoder 后: last_hidden_state shape={hidden_states.shape}")

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def linear_interpolation(features, seq_len):
    """
    Transpose the features to interpolate linearly.

    Args:
        features (torch.Tensor): The extracted features to be interpolated.
        seq_len (torch.Tensor): The sequence lengths of the features.

    Returns:
        torch.Tensor: The interpolated features.
    """
    print(f"[AUDIO_DEBUG] linear_interpolation(): 输入 shape={features.shape}, target seq_len={seq_len}")
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode='linear')
    output_features = output_features.transpose(1, 2)
    print(f"[AUDIO_DEBUG]   interpolation 输出: shape={output_features.shape}")
    return output_features
