import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.layers import Input, Lambda, Flatten, Reshape, Embedding, Concatenate
from tensorflow.keras.models import Model


class ClassToken(Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.class_token = tf.Variable(
            tf.zeros(shape=(1, 1, embed_dim)), trainable=True)

    def call(self, inputs):
        class_token = tf.broadcast_to(
            self.class_token, [tf.shape(inputs)[0], 1, self.embed_dim])
        return class_token

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
        })
        return config


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.mlp = self.build_mlp()
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def build_mlp(self):
        inputs = Input(shape=(None, self.embed_dim))
        x = Dense(self.mlp_dim, activation='gelu')(inputs)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(self.embed_dim)(x)
        x = Dropout(self.dropout_rate)(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def call(self, inputs):
        norm1_output = self.norm1(inputs)
        # Multi-head attention
        attn_output = self.mha(norm1_output, norm1_output)
        # Drop Out
        attn_output = self.dropout1(attn_output)
        # Add and Normalize
        norm2_output = Add()([norm1_output, attn_output])
        # MLP
        mlp_output = self.mlp(norm2_output)
        # Add, normalize and return the output.
        return Add()([norm2_output, mlp_output])

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


def create_vit(input_shape, patch_size, num_patches, num_classes, embed_dim, num_heads, mlp_dim, num_layers, dropout_rate):
    inputs = Input(shape=input_shape)

    print(f"Input shape: {inputs.shape}")
    print(f"No. of patches: {num_patches}")

#     # Patch to embedding
#     flatten_inputs = Flatten()(inputs)
#     patch_embed = Dense(embed_dim)(flatten_inputs)
#     patch_embed = tf.reshape(patch_embed, (-1, num_patches, embed_dim))

#     print(f"Patch Embed shape: {patch_embed.shape}")

# #    Patch creation and projection
#     patch_proj = []
#     for i in range(num_patches):
#         patch = Lambda(
#             lambda x: x[:, i*patch_size:(i+1)*patch_size, :])(inputs)
#         flattened_patch = Flatten()(patch)
#         projected_patch = Dense(embed_dim)(flattened_patch)
#         patch_proj.append(projected_patch)

#     patch_embed = tf.stack(patch_proj, axis=1)

    # Patch to embedding
    patch_embed = Conv1D(
        filters=embed_dim, kernel_size=patch_size, strides=patch_size)(inputs)

    # Calculate the number of patches
    num_patches = patch_embed.shape[1]

    print(f"Num Patches: {num_patches}")
    print(f"Patch Embed shape: {patch_embed.shape}")

    # Positional embeddings
    positional_embed_layer = Embedding(
        input_dim=num_patches, output_dim=embed_dim)
    # Linearly interpolate the position embeddings at different positions
    positions = tf.range(start=0, limit=num_patches, delta=1)
    print(f"Positions shape: {positions.shape}")
    # Reshape the position embeddings to match the shape of the inputs
    positional_embeddings = positional_embed_layer(positions)
    print(f"Positional Embed shape: {positional_embeddings.shape}")
    # Add the positional embeddings to the patch embeddings
    embed = patch_embed + positional_embeddings
    print(f"Embed shape: {embed.shape}")

    # Class token
    class_token = ClassToken(embed_dim)(embed)
    x = Concatenate(axis=1)([class_token, embed])

    # Transformer blocks
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, mlp_dim, dropout_rate)(x)

    # Classification head
    # x = tf.squeeze(x[:, 0:1], axis=1)  # Extract the class token representation
    x = x[:, 0]  # Only take the class token output
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


