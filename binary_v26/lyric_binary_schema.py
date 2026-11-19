"""Shared labels and dataset defaults for binary v26 workflows."""

PIPELINE_VERSION = "v26_binary"
CLASS_NAMES = ("Safe", "Explicit")
CLASS_IDS = tuple(range(len(CLASS_NAMES)))
NUM_CLASSES = len(CLASS_NAMES)
ID2LABEL = dict(enumerate(CLASS_NAMES))
LABEL2ID = {label: index for index, label in ID2LABEL.items()}
LABEL_NORMALIZATION = {label.casefold(): label for label in CLASS_NAMES}
DATASETS = ("Training_Cleaned_binary.csv", "Validation_Cleaned_binary.csv", "Testing_Cleaned_binary.csv")
