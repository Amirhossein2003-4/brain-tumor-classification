Tumor, context, or geometry? A patient-disjoint audit of brain-tumor MRI classifiers

A small, fully reproducible study on a public brain-tumor MRI dataset (3 tumor types). It asks three questions:

How much does the way you split the data inflate the score? (random slices vs. whole patients)
What does the classifier actually need: the tumor, the tissue around it, or just where the tumor is and what shape it has?
Does that answer differ between meningioma, glioma and pituitary tumors?

Everything ran on a CPU-only laptop (Intel i5-1334U, 16 GB RAM) with one small network (MobileNetV3-Small).

Research prototype. Not a medical device. No clinical claims. The findings are exploratory: the hypotheses were formed on the same 233 patients that were used to test them, there is no external test set, and most results come from one training seed. See What this does not show.

model overview

TL;DR
Finding	Evidence	Strength
A random split over slices leaks patients: 99.3 to 100 % of the test slices had a slice of the same patient in training. Splitting by patient fixes this by construction.	leakage report, 5 random folds	solid (it is a count)
For the same network and recipe, accuracy was 0.976 (random split) vs 0.913 (patient split) and calibration error rose from 0.012 to 0.054.	fold 1 of 5 only	preliminary
Trained and tested patient-disjoint over 5 folds, the full slice reaches macro AUC 0.971, the slice with the tumor removed 0.948, and tumor only 0.916.	233 patients, bootstrap over patients	moderate
Removing pixels at test time badly underestimates what a model can do without them: tumor-only scored 0.38 balanced accuracy when removed at test time, but 0.77 when the model was retrained on tumor-only input.	fold 1 of 5	preliminary, but it is the kind of effect ROAR-style evaluations warn about
The three tumor types differ in what they lean on: pituitary is largely explained by tumor location and outline, meningioma by the tumor's own tissue, glioma by both.	frozen-feature screening + 5-fold fine-tuning	exploratory, needs confirmation
Signal from real tissue structure only shows up far from the tumor (50 to 100 px, about 25 to 50 mm); right next to it, shuffling the pixels changes nothing.	ring experiment, frozen features	exploratory
Data
Dataset: Cheng et al., brain tumor dataset on figshare (article 1512427, https://doi.org/10.6084/m9.figshare.1512427). Check the terms on the figshare page before reuse. Please cite the dataset paper (see Related work).
Content: 3,064 T1-weighted contrast-enhanced slices from 233 patients: meningioma 708 slices (82 patients), glioma 1,426 (89), pituitary 930 (62). 1 to 38 slices per patient (median 13). Each slice has a hand-drawn tumor mask and a patient ID.
Things worth knowing: slices come from three anatomical planes mixed together with no plane label; two hospitals contributed but the center is not labeled; slice thickness is 6 mm (+1 mm gap), so neighboring slices of one patient look very alike; 15 slices are 256x256, the rest 512x512.
The dataset authors' own 5-fold file (cvind.mat) is patient-disjoint (0 of 233 patients appear in more than one fold). The leakage in many reproductions therefore comes from re-splitting slices at random, not from the dataset's official split.
What is in the repo

All scripts are single files with a CONFIG block (or key=value command-line overrides) at the top. Paths are Windows paths from my machine, so edit them first (see Reproduce).

Script	What it does
prep_data.py	.mat (HDF5 v7.3) to 16-bit PNG + masks + meta.csv; also checks whether the authors' folds keep patients together
export_view.py, show_inputs.py, show_bands.py	human-viewable copies and previews of every input condition
splits.py	the two split types (random slices vs. grouped by patient), the inner validation split (same type as the outer split), and a per-fold leakage report
data.py	per-image intensity normalization, resized cache (160 and 224 px), input conditions: full, tumor_only, context_only, mask_only, and band (distance rings around the tumor boundary, keep/remove, shuffled twins)
train.py	MobileNetV3-Small fine-tuning: partial freezing, BatchNorm adaptation, AdamW + one-cycle LR, per-epoch logs, per-block gradient reports, full checkpoints, final metrics and plots; one folder per run
summarize_runs.py	pools out-of-fold predictions, per-class AUC/recall, patient-level bootstrap CIs, paired contrasts between inputs, contrast heterogeneity across classes, real-vs-shuffled band contrasts, optional scan-plane strata
mask_baseline.py	geometry-only random forest on 13 handcrafted mask features (size, shape, location)
estimate_plane.py, label_planes.py	scan-plane estimation: ~145 hand labels + logistic regression on frozen embeddings, with cross-validated accuracy
screen_bands.py, band_presets.py	fast screening of dozens of input conditions with frozen embeddings + logistic regression and paired patient-level bootstrap
eval_ablation.py	test-time ablation of an already trained model (for comparison with retraining)
make_linkedin_images.py	builds picture cards from real slices (optional)
Methods in short
Model: ImageNet-pretrained MobileNetV3-Small (0.93 M parameters). Input: grayscale slice, per-image percentile normalization (1st to 99.5th), resized to 160x160, replicated to 3 channels. The first 4 blocks (strides 2, 4, 8) are frozen (10,488 parameters, 1.1 %), the other 9 blocks and a new head (global average pool, dropout 0.2, linear 576 to 3) are fine-tuned (918,251 parameters). BatchNorm layers of frozen blocks stay in train mode so their statistics adapt to MRI intensities.
Training: AdamW (weight decay 0.01), one-cycle LR (peak 5e-4 backbone, 2e-3 head), class-weighted cross-entropy, batch 32, 30 epochs, no augmentation. Best epoch picked on a validation split built the same way as the test split. About 25 to 50 minutes per run on the laptop CPU.
Splits: StratifiedKFold over slices (random) or StratifiedGroupKFold over patient IDs (patient-disjoint). 5 folds, seed 0.
Uncertainty: slices of one patient are not independent, so intervals come from a bootstrap that resamples patients (within each class), with the same resampled patients for every model so differences are paired.
Input conditions: full; tumor_only (original pixels inside the mask, black elsewhere); context_only (mask region filled with a smooth blur); mask_only (binary outline); band (keep or remove a ring at a given distance from the tumor boundary, optionally with the ring's pixels shuffled).
Results
1. Splitting by slice leaks patients (fold 1 of 5)
	random split	patient split
train / val / test slices	2144 / 307 / 613	2169 / 343 / 552
test patients (also present in train)	195 (194)	46 (0)
test slices with a same-patient slice in train	612 / 613	0 / 552
accuracy	0.9755	0.9130
balanced accuracy	0.9701	0.9158
macro F1	0.9718	0.8926
ROC AUC (macro, OvR)	0.9981	0.9826
calibration error (ECE)	0.0115	0.0544

Across the five random folds, 99.3 to 100 % of test slices had a same-patient slice in training. The gap above comes from a single fold and a single seed, and the two test sets are not identical (613 slices from 195 patients vs. 552 slices from 46 patients), so read it as an illustration, not an effect size. The 5-fold repeat of the random-split runs is not done.

2. What does the model need? 5-fold patient-disjoint results

Pooled out-of-fold predictions, 233 patients, 95 % CI from a patient-level bootstrap.

input	AUC meningioma	AUC glioma	AUC pituitary	macro AUC	balanced acc	macro F1	ECE
full slice	0.949 [0.921, 0.971]	0.975 [0.960, 0.986]	0.988 [0.976, 0.997]	0.971 [0.956, 0.983]	0.880	0.878	0.071
tumor hidden (context_only)	0.909 [0.869, 0.943]	0.953 [0.929, 0.972]	0.983 [0.966, 0.995]	0.948 [0.927, 0.967]	0.840	0.838	0.113
tumor only	0.905 [0.877, 0.931]	0.910 [0.881, 0.934]	0.933 [0.913, 0.951]	0.916 [0.895, 0.935]	0.762	0.762	0.148

Paired AUC differences (95 % CI):

contrast	meningioma	glioma	pituitary	macro
full − tumor hidden	+0.040 [0.023, 0.059]	+0.022 [0.011, 0.035]	+0.005 [0.001, 0.010]	+0.022 [0.013, 0.033]
full − tumor only	+0.044 [0.019, 0.069]	+0.065 [0.039, 0.095]	+0.055 [0.038, 0.073]	+0.055 [0.036, 0.075]
tumor hidden − tumor only	+0.004 [−0.030, 0.036]	+0.044 [0.012, 0.076]	+0.049 [0.032, 0.069]	+0.032 [0.010, 0.057]

The usefulness of the context differs between classes (differences of the last row: glioma − meningioma +0.039 [0.014, 0.066]; meningioma − pituitary −0.045 [−0.078, −0.014]). These comparisons were not pre-registered and there are many of them, so treat the intervals as hints. Per-fold balanced accuracy varies by about ±0.03.

3. Removing pixels at test time vs. retraining without them (fold 1 of 5, patient split)
input shown to the model	balanced accuracy, test-time removal from the full-image model	balanced accuracy, model retrained on that input
full	0.916	0.916
tumor hidden	0.802	0.897
tumor only	0.379	0.767

A model never shown a mostly black image collapses when shown one. This is a distribution-shift effect, so test-time ablation alone says little about how informative a region is.

4. How much does geometry alone say? (13 mask features + random forest, 5 folds)
split	accuracy	balanced accuracy	macro F1
random slices	0.804 ± 0.009	0.781 ± 0.008	0.780 ± 0.008
by patient	0.698 ± 0.040	0.664 ± 0.041	0.652 ± 0.049

The shape and position of the mask alone are far above chance (0.33 balanced accuracy). The gap between the two splits (about 0.11 to 0.12) appears even for a model that sees no image at all.

5. Where in the context does the signal live? (frozen-feature screening)

Each condition is embedded once with the frozen pretrained network and a logistic regression is trained with the same patient-disjoint folds (minutes instead of hours). This ranks conditions by linearly decodable information. It is a screening tool: rankings can change after fine-tuning. Macro AUC, 233 patients, 1,000 patient-level bootstrap draws:

condition	macro AUC [95 % CI]	men	gli	pit
full	0.947 [0.929, 0.962]	0.907	0.947	0.986
tumor only	0.927 [0.911, 0.942]	0.931	0.914	0.936
tumor hidden	0.933 [0.911, 0.952]	0.879	0.937	0.984
outline only (silhouette)	0.808 [0.784, 0.832]	0.712	0.784	0.929
ring 0 to 10 px (real / shuffled)	0.846 / 0.848			
ring 10 to 25 px (real / shuffled)	0.846 / 0.834			
ring 25 to 50 px (real / shuffled)	0.841 / 0.816			
ring 50 to 100 px (real / shuffled)	0.891 / 0.817			
ring 100 to 200 px	0.914	0.841	0.911	0.991
tumor + ring to 100 px	0.959 [0.946, 0.971]	0.926	0.961	0.991
tumor kept, 0 to 100 px ring removed, rest kept	0.967 [0.953, 0.977]	0.936	0.972	0.993

(Distances are in pixels of the original 512 grid; at the dataset's stated 0.49 mm per pixel, 10 px is about 5 mm and 100 px about 49 mm. 15 slices are 256x256, so this is approximate.)

Paired contrasts that stood out:

tumor only − outline only: meningioma +0.219 [0.179, 0.258], glioma +0.130 [0.099, 0.163], pituitary +0.007 [−0.004, 0.017]. Tissue inside the tumor adds a lot for meningioma and glioma and nothing for pituitary, whose label is almost fully recoverable from the tumor outline and location.
real ring − shuffled twin (same geometry and histogram): 0 to 10 px: −0.001 [−0.010, 0.007]; 25 to 50 px: +0.025 [0.009, 0.040]; 50 to 100 px: +0.074 [0.053, 0.096]. Structure in the tissue matters only farther from the tumor.
tumor + ring to 100 px − tumor only: meningioma −0.006, glioma +0.047 [0.024, 0.071], pituitary +0.055 [0.039, 0.073]. Adding surroundings helps glioma and pituitary, not meningioma.
6. Scan plane (estimated, not in the dataset)

Plane labels do not exist. I labeled 144 random slices by eye (47 axial, 49 coronal, 48 sagittal) and trained a logistic regression on frozen embeddings: patient-grouped cross-validated accuracy 0.951 (about 0.92 to 0.99 with 144 labels). Predicted counts were about a third each. The association between plane and tumor class is weak (Cramer's V about 0.05; meningioma 31/31/39 % axial/coronal/sagittal, glioma 36/34/31 %, pituitary 31/35/34 %), so the plane alone is unlikely to be a strong label shortcut. Slices were not evaluated plane by plane for all conditions yet.

What this does not show
No causal claims. The experiments compare models trained on different input representations. They do not measure how a network "really" decides, and they do not say which tissue is biologically informative.
Masking is not neutral. A removed tumor leaves a hole that still reveals the tumor's location and shape. keep rings reveal the ring's geometry. The shuffled twins, the outline-only input and the geometry baseline partially control this; they do not remove it.
Tumor-only is a weak input here: it is about 98 % black and the tumor covers about 1.7 % of the image, so after resizing from 512 to 160 px little texture is left. Some of "context beats tumor" can be a representation effect. A native-resolution tumor crop is the next control.
Frozen-feature screening is not fine-tuning. Fine-tuning helped the full and context inputs more than the tumor-only input in this data.
One dataset, two hospitals (unlabeled), 233 patients, 2D slices, one architecture, mostly one seed. The per-class contrasts rest on 62 to 89 patients per class.
Everything is exploratory. The cleanest next step is to freeze the hypotheses and thresholds, then confirm on new seeds and an independent dataset (many public brain-MRI sets are derived from this one, so provenance must be checked).
A pipeline lesson (found on a synthetic dry run): torchvision's MobileNetV3 uses BatchNorm momentum 0.01. With short training, running statistics lag far behind and validation can look random while training accuracy is perfect. The code sets 0.1 by default (bn_momentum). I did not test the real data with 0.01.
Related work (not a systematic review)

Closest prior work I found, so you can judge novelty yourself. Details were read from abstracts or short excerpts, except where noted.

Cheng et al., 2015, Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition, PLOS ONE. The dataset paper; compares the tumor region, an enlarged region and ring-shaped partitions, and notes class-specific error changes. https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0140381
Mask of Truth: Model Sensitivity to Unexpected Regions of Medical Images (J Imaging Inform Med, 2025). Trains models with and without the region of interest in chest X-ray and fundus images. https://arxiv.org/abs/2412.04030
Hooker et al., A Benchmark for Interpretability Methods in Deep Neural Networks (ROAR; remove and retrain). https://arxiv.org/abs/1806.10758
Xiao et al., Noise or Signal: The Role of Image Backgrounds in Object Recognition (ICLR 2021). https://arxiv.org/abs/2006.09994
Vangala et al., 2026 preprint on leakage and shortcuts in public brain-tumor MRI benchmarks (patient overlap between train and test in Kaggle-style corpora). https://arxiv.org/abs/2610.00421 (recent preprint; I have not seen a peer-reviewed version)
Comparison of cropped, uncropped and segmented lesion inputs on the same 3-class dataset. https://arxiv.org/abs/2001.08844
Intratumoral vs. peritumoral radiomics for brain-tumor differentiation (Frontiers in Oncology, 2022). https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2022.848846/full

I did not find a study that crosses patient-disjoint evaluation with retrained tumor/context/outline inputs per tumor class, but absence from my search is not evidence of novelty.

Reproduce
bash
pip install torch torchvision scikit-learn scipy h5py pillow matplotlib numpy
Download the dataset from figshare and unzip the four archives.
Edit the paths at the top of the scripts (prep_data.py: SRC and DST; splits.py: ROOT; train.py: run_root and pretrained_path; plus the two paths in screen_bands.py and estimate_plane.py). The scripts expect a local copy of the torchvision MobileNetV3-Small weights and fall back to downloading them.
Run, from the repo folder:
bash
python prep_data.py                    # .mat -> PNG + meta.csv (+ checks the authors' folds)
python splits.py                       # leakage report for both split types
python data.py 160 224                 # build the resized caches
python mask_baseline.py                # geometry-only baseline + out-of-fold predictions

# fine-tuning: one run = one (split, fold, input); 30 epochs, roughly 25 to 50 min on a laptop CPU
python train.py mode=patient fold=0 epochs=30 input_mode=full
python train.py mode=patient fold=0 epochs=30 input_mode=tumor_only
python train.py mode=patient fold=0 epochs=30 input_mode=context_only
#   ... repeat for folds 1 to 4, and mode=slice for the leakage comparison

python summarize_runs.py epochs=30     # pooled out-of-fold metrics, patient-level bootstrap CIs, contrasts

python screen_bands.py preset=full     # frozen-feature screening of ~23 input conditions (minutes)
python estimate_plane.py && python label_planes.py n=150   # scan-plane estimate

Every run writes its own folder (config, per-epoch log, per-block gradient reports, best.pth/last.pth full checkpoints, predictions, metrics, plots).

AI assistance

I designed the questions and ran every experiment myself, and I used AI assistants (Claude and ChatGPT) for code drafting and literature search. Code was tested on synthetic data before use.

License and citation

Code: add a LICENSE file of your choice (for example MIT). Data: see the figshare page; please cite Cheng et al. (2015) when using the dataset.# brain-tumor-classification
