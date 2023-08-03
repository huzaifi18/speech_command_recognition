MAIN_PATH = "../dataset/noisy/test/"

# IDN
# labels = ["atas", "bawah", "berhenti", "hidup", "iya", "kanan", "kiri", "mati", "pergi", "tidak"]

# ENG
labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown"]

IDN_DATASET = False
sr = 8000
return_df = False

n_fft = 512

# Parameter for MFCC extraction
n_mfcc = 15
MFCC_PATH_OUTPUT = f"../dataset/pickle/test/mfcc_{n_mfcc}_ENG_noisy.pkl"
LABEL_MFCC_PATH_OUTPUT = f"../dataset/pickle/test/label_mfcc_{n_mfcc}_ENG_noisy.pkl"
CSV_MFCC_OUTPUT = f"../dataset/mfccs_{n_mfcc}_IDN.csv"

# Parameter for FBANK extraction
FBANK_PATH_OUTPUT = "../dataset/pickle/test/fbank_ENG_noisy.pkl"
LABEL_FBANK_PATH_OUTPUT = "../dataset/pickle/test/label_fbank_ENG_noisy.pkl"
CSV_FBANK_OUTPUT = f"../dataset/fbank_IDN.csv"

# Parameter for PNCC extraction
n_pncc = 15
PNCC_PATH_OUTPUT = f"../dataset/pickle/test/pncc_{n_pncc}_ENG_noisy.pkl"
LABEL_PNCC_PATH_OUTPUT = f"../dataset/pickle/test/label_pncc_{n_pncc}_ENG_noisy.pkl"
CSV_PNCC_OUTPUT = f"../dataset/pncc_{n_pncc}_ENG.csv"