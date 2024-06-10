import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.layers import Input, Reshape, Embedding, Concatenate
from tensorflow.keras.models import Model


class ClassToken(Layer):
    def __init__(self, embed_dim):
        super(ClassToken, self).__init__()
        self.class_token = tf.Variable(
            tf.random.normal((1, 1, embed_dim)), trainable=True)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        class_token = tf.repeat(self.class_token, repeats=batch_size, axis=0)
        return class_token


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

        self.dense1 = Dense(mlp_dim, activation="gelu")
        self.dropout1 = Dropout(dropout_rate)
        self.dense2 = Dense(embed_dim)
        self.dropout2 = Dropout(dropout_rate)

    def mlp(self, x, training):
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return x

    def call(self, inputs, training):
        # Multi-head attention
        attn_output = self.mha(inputs, inputs)
        # Drop Out
        attn_output = self.dropout1(attn_output, training=training)
        # Add and Normalize
        norm1_output = self.norm1(inputs + attn_output)
        # MLP
        mlp_output = self.mlp(norm1_output, training)
        # Add, normalize and return the output.
        return self.norm2(norm1_output + mlp_output)


def create_vit(input_shape, patch_size, num_patches, num_classes, embed_dim, num_heads, mlp_dim, num_layers, dropout_rate):
    inputs = Input(shape=input_shape)
    inputs = Reshape((-1, input_shape[-1]))(inputs)

    print(f"Input shape: {inputs.shape}")
    print(f"No. of patches: {num_patches}")

    # Patch to embedding
    patch_embed = Dense(embed_dim)(inputs)

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
    x = tf.squeeze(x[:, 0:1], axis=1)  # Extract the class token representation
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    return model
