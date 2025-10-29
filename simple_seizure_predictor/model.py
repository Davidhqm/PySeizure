"""
CNN-Transformer model for seizure prediction
Based on Koutsouvelis et al. 2024
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class CNNTransformerSeizurePredictor(keras.Model):
    """
    CNN-Transformer architecture for seizure prediction

    Architecture:
    - 3 CNN layers for spatial-temporal feature extraction
    - Transformer layers for long-term dependencies
    - Binary classification output
    """

    def __init__(self,
                 input_shape,
                 cnn_filters=[32, 64, 128],
                 kernel_size=(3, 3),
                 pool_size=(2, 2),
                 dropout_cnn=0.1,
                 num_heads=8,
                 key_dim=64,
                 num_transformer_blocks=2,
                 ff_dim=64,
                 dropout_transformer=0.3,
                 **kwargs):
        super().__init__(**kwargs)

        self.input_shape_model = input_shape

        # CNN Layers
        self.cnn_blocks = []
        for i, filters in enumerate(cnn_filters):
            block = {
                'conv': layers.Conv2D(filters, kernel_size, padding='same',
                                     activation=None, name=f'conv_{i}'),
                'bn': layers.BatchNormalization(name=f'bn_{i}'),
                'activation': layers.ReLU(name=f'relu_{i}'),
                'dropout': layers.Dropout(dropout_cnn, name=f'dropout_cnn_{i}'),
                'pool': layers.MaxPooling2D(pool_size, name=f'pool_{i}')
            }
            self.cnn_blocks.append(block)

        # Reshape for transformer
        self.reshape_layer = layers.Reshape((-1, cnn_filters[-1]))

        # Transformer blocks
        self.transformer_blocks = []
        for i in range(num_transformer_blocks):
            block = {
                'attention': layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=key_dim,
                    dropout=dropout_transformer,
                    name=f'attention_{i}'
                ),
                'dense': layers.Dense(ff_dim, activation='relu', name=f'ff_dense_{i}'),
                'dropout': layers.Dropout(dropout_transformer, name=f'dropout_trans_{i}'),
            }
            self.transformer_blocks.append(block)

        # Classification head
        self.flatten = layers.Flatten()
        self.output_dense = layers.Dense(1, activation='sigmoid', name='output')

    def call(self, inputs, training=False):
        """Forward pass"""
        # Add channel dimension if needed: (batch, time, channels) -> (batch, time, channels, 1)
        if len(inputs.shape) == 3:
            x = tf.expand_dims(inputs, axis=-1)
        else:
            x = inputs

        # CNN feature extraction
        for block in self.cnn_blocks:
            x = block['conv'](x)
            x = block['bn'](x, training=training)
            x = block['activation'](x)
            x = block['dropout'](x, training=training)
            x = block['pool'](x)

        # Reshape: (batch, time, channels, features) -> (batch, time*channels, features)
        batch_size = tf.shape(x)[0]
        time_dim = tf.shape(x)[1]
        channel_dim = tf.shape(x)[2]
        feature_dim = tf.shape(x)[3]

        # Merge time and channel dimensions
        x = tf.reshape(x, (batch_size, time_dim * channel_dim, feature_dim))

        # Transformer layers
        for block in self.transformer_blocks:
            # Multi-head attention
            attn_output = block['attention'](x, x, training=training)

            # Feed-forward
            ff_output = block['dense'](attn_output)
            ff_output = block['dropout'](ff_output, training=training)

            x = ff_output

        # Classification
        x = self.flatten(x)
        output = self.output_dense(x)

        return output

    def build_model(self):
        """Build model to see architecture"""
        inputs = keras.Input(shape=self.input_shape_model)
        outputs = self.call(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs, name='CNNTransformerSeizurePredictor')
        return model


def create_model(input_shape, config=None):
    """
    Create and compile CNN-Transformer model

    Args:
        input_shape: (time_steps, n_channels) e.g., (1280, 23) for 5-sec at 256Hz with 23 channels
        config: Model configuration dictionary

    Returns:
        Compiled Keras model
    """
    if config is None:
        config = {
            'cnn_filters': [32, 64, 128],
            'kernel_size': (3, 3),
            'pool_size': (2, 2),
            'dropout_cnn': 0.1,
            'num_heads': 8,
            'key_dim': 64,
            'num_transformer_blocks': 2,
            'ff_dim': 64,
            'dropout_transformer': 0.3,
        }

    model = CNNTransformerSeizurePredictor(input_shape=input_shape, **config)

    # Compile with Adam optimizer and binary cross-entropy loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, amsgrad=True),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )

    return model


def get_callbacks(patience=20, checkpoint_path='best_model.h5'):
    """Create training callbacks"""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]

    return callbacks
