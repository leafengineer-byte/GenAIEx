import os
import io
import base64

import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "vae_encoder.keras"
)

DECODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "vae_decoder.keras"
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

print("Loading VAE models...")

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(
        shape=tf.shape(z_mean)
    )
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def load_keras_model(path):
    dense_layer = tf.keras.layers.Dense
    original_init = dense_layer.__init__

    def compatible_init(self, *args, **kwargs):
        if kwargs.get("quantization_config") is None:
            kwargs.pop("quantization_config", None)
        original_init(self, *args, **kwargs)

    dense_layer.__init__ = compatible_init
    try:
        return tf.keras.models.load_model(
            path,
            compile=False,
            custom_objects={"sampling": sampling}
        )
    finally:
        dense_layer.__init__ = original_init


encoder = load_keras_model(ENCODER_PATH)
decoder = load_keras_model(DECODER_PATH)

print("Encoder loaded successfully.")
print("Decoder loaded successfully.")


# ============================================================
# IMAGE -> BASE64
# ============================================================

def image_to_base64(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# PREPROCESS MNIST IMAGE
# ============================================================

def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # Resize
    image = image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    # Convert to numpy
    img = np.array(
        image
    ).astype("float32") / 255.0

    # Invert if background is white
    if img.mean() > 0.5:
        img = 1.0 - img

    return img


# ============================================================
# CREATE LATENT SPACE VISUALIZATION
# ============================================================

def create_latent_visualization(
    latent_point
):

    # Create a grid around the input point
    grid_size = 15

    x_values = np.linspace(
        -3,
        3,
        grid_size
    )

    y_values = np.linspace(
        -3,
        3,
        grid_size
    )

    X, Y = np.meshgrid(
        x_values,
        y_values
    )

    points = np.column_stack(
        (
            X.ravel(),
            Y.ravel()
        )
    )

    generated = decoder.predict(
        points,
        verbose=0
    )

    # Calculate image brightness
    brightness = generated.mean(
        axis=(1, 2, 3)
    )

    plt.figure(
        figsize=(7, 7)
    )

    scatter = plt.scatter(
        points[:, 0],
        points[:, 1],
        c=brightness,
        cmap="viridis",
        s=35,
        alpha=0.65
    )

    # Highlight current input
    plt.scatter(
        latent_point[0],
        latent_point[1],
        s=300,
        marker="*",
        color="red",
        edgecolors="white",
        linewidths=2,
        label="Input Latent Position"
    )

    plt.axhline(
        0,
        color="gray",
        linewidth=0.7,
        alpha=0.5
    )

    plt.axvline(
        0,
        color="gray",
        linewidth=0.7,
        alpha=0.5
    )

    plt.xlabel(
        "Latent Dimension 1 (z₁)"
    )

    plt.ylabel(
        "Latent Dimension 2 (z₂)"
    )

    plt.title(
        "2D VAE Latent Space"
    )

    plt.legend()

    plt.grid(
        alpha=0.15
    )

    plt.tight_layout()

    buffer = io.BytesIO()

    plt.savefig(
        buffer,
        format="png",
        dpi=140,
        transparent=True
    )

    plt.close()

    buffer.seek(0)

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# GENERATE RANDOM IMAGES
# ============================================================

@app.route(
    "/generate",
    methods=["POST"]
)
def generate():

    try:

        count = int(
            request.json.get(
                "count",
                9
            )
        )

        count = max(
            1,
            min(count, 25)
        )

        # Random latent vectors
        latent_vectors = np.random.normal(
            size=(count, 2)
        )

        # Generate images
        generated = decoder.predict(
            latent_vectors,
            verbose=0
        )

        images = []

        for i in range(count):

            arr = (
                generated[i].squeeze()
                * 255
            ).clip(
                0,
                255
            ).astype(
                np.uint8
            )

            image = Image.fromarray(
                arr
            )

            images.append(
                {
                    "image":
                        image_to_base64(
                            image
                        ),

                    "z1":
                        round(
                            float(
                                latent_vectors[i, 0]
                            ),
                            3
                        ),

                    "z2":
                        round(
                            float(
                                latent_vectors[i, 1]
                            ),
                            3
                        )
                }
            )

        return jsonify(
            {
                "success": True,
                "images": images
            }
        )

    except Exception as e:

        return jsonify(
            {
                "success": False,
                "error": str(e)
            }
        ), 500


# ============================================================
# RECONSTRUCT IMAGE
# ============================================================

@app.route(
    "/reconstruct",
    methods=["POST"]
)
def reconstruct():

    try:

        if "image" not in request.files:

            return jsonify(
                {
                    "success": False,
                    "error":
                        "No image uploaded."
                }
            ), 400

        file = request.files["image"]

        image = Image.open(
            file.stream
        )

        # Preprocess
        processed = preprocess_image(
            image
        )

        # Add batch dimension
        input_image = processed.reshape(
            1,
            28,
            28,
            1
        )

        # Encoder
        z_mean, z_log_var, z = encoder.predict(
            input_image,
            verbose=0
        )

        # Decoder
        reconstructed = decoder.predict(
            z,
            verbose=0
        )

        reconstructed_array = (
            reconstructed[0].squeeze()
            * 255
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )

        reconstructed_image = Image.fromarray(
            reconstructed_array
        )

        # Latent visualization
        latent_graph = create_latent_visualization(
            z_mean[0]
        )

        return jsonify(
            {
                "success": True,

                "input":
                    image_to_base64(
                        Image.fromarray(
                            (
                                processed * 255
                            ).astype(
                                np.uint8
                            )
                        )
                    ),

                "reconstruction":
                    image_to_base64(
                        reconstructed_image
                    ),

                "z1":
                    round(
                        float(
                            z_mean[0, 0]
                        ),
                        4
                    ),

                "z2":
                    round(
                        float(
                            z_mean[0, 1]
                        ),
                        4
                    ),

                "latent_graph":
                    latent_graph
            }
        )

    except Exception as e:

        return jsonify(
            {
                "success": False,
                "error": str(e)
            }
        ), 500


# ============================================================
# RUN FLASK
# ============================================================


@app.route('/decode_point', methods=['POST'])
def decode_point():
    try:
        data = request.json or {}
        z1 = float(data.get('z1', 0.0))
        z2 = float(data.get('z2', 0.0))
        pt = np.array([[z1, z2]], dtype=np.float32)
        generated = decoder.predict(pt, verbose=0)
        arr = (generated[0].squeeze() * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        return jsonify({'success': True, 'image': image_to_base64(img), 'z1': z1, 'z2': z2})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=int(os.environ.get('PORT', 5005))
    )
