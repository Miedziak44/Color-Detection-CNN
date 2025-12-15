import time
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from color_detection import config
from color_detection import dataset
from color_detection import model as model_module


# --- TF → numpy ---
def ds_to_numpy(ds, img_size=(32, 32)):
    X, y = [], []
    for images, labels in ds:
        imgs_resized = tf.image.resize(images, img_size).numpy()
        imgs_flat = imgs_resized.reshape(len(imgs_resized), -1)
        X.append(imgs_flat)
        y.append(labels.numpy())
    return np.vstack(X), np.concatenate(y)


def compute_weights(train_ds, class_names):
    print("Obliczanie class weights...")
    labels = np.concatenate([y for _, y in train_ds], axis=0)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    class_weight_dict = dict(enumerate(class_weights))

    # Zmniejszenie wagi klasy “black”
    if "black" in class_names:
        idx = class_names.index("black")
        class_weight_dict[idx] *= 0.4
        print("Skorygowano wagę klasy 'black'.")

    return class_weight_dict


def benchmark_full_training(NUM_EPOCHS=50, NUM_INFER=100):

    print("\n📥 Ładowanie datasetu z:", config.DATA_DIR)
    train_ds, val_ds = dataset.get_datasets()

    class_names = train_ds.class_names
    num_classes = len(class_names)

    # ============================================
    #    1) TRENING CNN (pełny, jak w projekcie)
    # ============================================
    print("\n======================")
    print("  Benchmark: TensorFlow CNN (pełne trenowanie)")
    print("======================")

    model = model_module.build_model(num_classes)

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    # class weights
    class_weights = compute_weights(train_ds, class_names)

    print(f"Trenowanie CNN przez {NUM_EPOCHS} epok...")

    t_train_start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS,
        class_weight=class_weights,
        verbose=1
    )
    tf_train_time = time.time() - t_train_start

    cnn_acc = history.history["val_accuracy"][-1]

    # inference benchmark
    for batch in val_ds.take(1):
        samples = batch[0][:NUM_INFER]
        break

    t0 = time.time()
    model.predict(samples, verbose=0)
    cnn_infer_per_img = (time.time() - t0) / NUM_INFER

    print(f"\n🔵 CNN wyniki:")
    print(f"  ✔ Val accuracy: {cnn_acc:.4f}")
    print(f"  ⏱ Training time: {tf_train_time:.3f}s")
    print(f"  ⚡ Inference time: {cnn_infer_per_img * 1000:.3f} ms / image")


    # ============================================
    #    2) RANDOM FOREST
    # ============================================
    print("\n======================")
    print("  Benchmark: RandomForest (scikit-learn)")
    print("======================")

    print("Konwersja datasetu TF → numpy...")
    X_train, y_train = ds_to_numpy(train_ds)
    X_val, y_val = ds_to_numpy(val_ds)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1
    )

    print("Trenowanie RandomForest...")
    t0 = time.time()
    clf.fit(X_train, y_train)
    rf_train_time = time.time() - t0

    preds = clf.predict(X_val)
    rf_acc = accuracy_score(y_val, preds)

    # inference
    t0 = time.time()
    clf.predict(X_val[:NUM_INFER])
    rf_infer_per_img = (time.time() - t0) / NUM_INFER

    print(f"\n🟢 RandomForest:")
    print(f"  ✔ Val accuracy: {rf_acc:.4f}")
    print(f"  ⏱ Training time: {rf_train_time:.3f}s")
    print(f"  ⚡ Inference time: {rf_infer_per_img * 1000:.3f} ms / image")


    # ============================================
    #    PODSUMOWANIE
    # ============================================
    print("\n==============================================")
    print("                 TEST SUMMARY")
    print("==============================================")
    print(f"TensorFlow CNN accuracy:     {cnn_acc:.4f}")
    print(f"RandomForest accuracy:       {rf_acc:.4f}")
    print("----------------------------------------------")
    print(f"TF training ({NUM_EPOCHS} epochs): {tf_train_time:.3f}s")
    print(f"RF training:                 {rf_train_time:.3f}s")
    print("----------------------------------------------")
    print(f"TF inference per image:      {cnn_infer_per_img * 1000:.3f} ms")
    print(f"RF inference per image:      {rf_infer_per_img * 1000:.3f} ms")
    print("==============================================\n")


if __name__ == "__main__":
    benchmark_full_training()
