import os
import numpy as np
from scipy.interpolate import interp1d
import string

# -------------------------
# CONFIG
# -------------------------
for letter in string.ascii_uppercase:
    INPUT_DIR = f"dataset/{letter}"
    OUTPUT_DIR = f"dataset_augmented/{letter}"

    AUGS_PER_SAMPLE = 5
    TARGET_FRAMES = 30

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    # -------------------------
    # AUGMENTATIONS
    # -------------------------

    def add_noise(data, std=0.01):
        noise = np.random.normal(0, std, data.shape)
        return data + noise


    def random_scale(data, scale_range=(0.95, 1.05)):
        scale = np.random.uniform(*scale_range)

        result = data.copy()

        for hand in range(21):
            result[:, hand*3] *= scale
            result[:, hand*3+1] *= scale

        return result


    def random_translate(data, max_shift=0.03):
        tx = np.random.uniform(-max_shift, max_shift)
        ty = np.random.uniform(-max_shift, max_shift)

        result = data.copy()

        for hand in range(21):
            result[:, hand*3] += tx
            result[:, hand*3+1] += ty

        return result


    def temporal_warp(sequence):
        old_idx = np.arange(len(sequence))

        speed = np.random.uniform(0.9, 1.1)

        new_length = max(5, int(len(sequence) * speed))

        new_idx = np.linspace(0, len(sequence)-1, new_length)

        interp = interp1d(
            old_idx,
            sequence,
            axis=0,
            kind="linear",
            fill_value="extrapolate"
        )

        warped = interp(new_idx)

        final_idx = np.linspace(
            0,
            len(warped)-1,
            TARGET_FRAMES
        )

        interp2 = interp1d(
            np.arange(len(warped)),
            warped,
            axis=0,
            kind="linear"
        )

        return interp2(final_idx)


    def augment(sample):
        x = sample.copy()

        if np.random.rand() < 0.8:
            x = add_noise(x)

        if np.random.rand() < 0.8:
            x = random_scale(x)

        if np.random.rand() < 0.8:
            x = random_translate(x)

        if np.random.rand() < 0.7:
            x = temporal_warp(x)

        return x


    # -------------------------
    # PROCESS
    # -------------------------

    counter = 0

    files = sorted(
        [f for f in os.listdir(INPUT_DIR)
        if f.endswith(".npy")]
    )

    for file in files:

        path = os.path.join(INPUT_DIR, file)

        sample = np.load(path)

        # save original
        np.save(
            os.path.join(
                OUTPUT_DIR,
                f"sample_{counter:04d}.npy"
            ),
            sample
        )

        counter += 1

        # save augmentations
        for _ in range(AUGS_PER_SAMPLE):

            aug = augment(sample)

            np.save(
                os.path.join(
                    OUTPUT_DIR,
                    f"sample_{counter:04d}.npy"
                ),
                aug
            )

            counter += 1

    print(f"Created {counter} samples.")