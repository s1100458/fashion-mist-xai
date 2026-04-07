import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

#Normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#Add channel dimension (28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print(x_train.shape, x_test.shape)


model = models.Sequential([
    layers.Input(shape=(28, 28, 1)), 
    
    layers.Conv2D(32, (3, 3), activation="relu"), 
    layers.MaxPooling2D((2, 2)), 

    layers.Conv2D(64, (3, 3), activation="relu"), 
    layers.MaxPooling2D((2, 2)), 

    layers.Flatten(),
    layers.Dense(64, activation="relu"), 
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

pred_probs = model.predict(x_test, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1)

correct_indices = np.where(pred_labels == y_test)[0]
incorrect_indices = np.where(pred_labels != y_test)[0]

print("Correct predictions:", len(correct_indices))
print("Incorrect predictions:", len(incorrect_indices))


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.outputs[0]]
    )

    with tf.GradientTape() as tape: 
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    

def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    image_x = tf.expand_dims(image, axis=0)
    return baseline_x + alphas_x * (image_x - baseline_x)

def compute_gradients(images, target_class_idx, model):
    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images)
        outputs = predictions[:, target_class_idx]
    return tape.gradient(outputs, images)

def integral_approximation(gradients):
    grads = (gradients[:-1] + gradients[1:]) / 2.0
    return tf.reduce_mean(grads, axis=0)

def integrated_gradients(model, image, target_class_idx, baseline=None, m_steps=50):
    if baseline is None:
        baseline = tf.zeros_like(image)

    alphas = tf.linspace(0.0, 1.0, m_steps + 1)
    interpolated_images = interpolate_images(baseline, image, alphas)
    grads = compute_gradients(interpolated_images, target_class_idx, model)
    avg_grads = integral_approximation(grads)

    integrated_grads = (image - baseline) * avg_grads
    return integrated_grads


if __name__ == "__main__":
    def plot_explanations(idx, title_prefix, save_path=None):
        img = x_test[idx]
        img_array = np.expand_dims(img, axis=0)

        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_1")

        ig_attributions = integrated_gradients(
            model,
            tf.convert_to_tensor(img),
            pred_labels[idx]
        )
        ig_map = tf.reduce_sum(tf.abs(ig_attributions), axis=-1).numpy()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"{title_prefix}\nTrue: {class_names[y_test[idx]]}\nPred: {class_names[pred_labels[idx]]}")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.imshow(heatmap, cmap="jet", alpha=0.4)
        plt.title("Grad-CAM")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(ig_map, cmap="hot")
        plt.title("Integrated Gradients")
        plt.axis("off")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    plot_explanations(correct_indices[0], "Correct Example", "correct_example.png")
    plot_explanations(incorrect_indices[0], "Incorrect Example", "incorrect_example.png")