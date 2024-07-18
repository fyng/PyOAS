SEQUENCE_BOS_TOKEN = 0
SEQUENCE_PAD_TOKEN = 1
SEQUENCE_EOS_TOKEN = 2
SEQUENCE_CHAINBREAK_TOKEN = 31
SEQUENCE_MASK_TOKEN = 32

MASK_STR = "<mask>"
BOS_STR = "<cls>"
EOS_STR = "<eos>"
PAD_STR = "<pad>"
UNK_STR = "<unk>"
CHAIN_BREAK_STR = "|"
GAP_STR = "."

SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>", "<mask>", ".", "|",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O"
]

VDJ_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>", "<mask>", ".", "|",
    "V", "D", "J"
]

ANARCI_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>", "<mask>", ".", "|",
    "fwk1", "fwk2", "fwk3", "fwk4",   
    "cdrk1", "cdrk2", "cdrk3", "cdrk4",
    "fwl1", "fwl2", "fwl3", "fwl4",   
    "cdrl1", "cdrl2", "cdrl3", "cdrl4",
    "fwh1", "fwh2", "fwh3", "fwh4",   
    "cdrh1", "cdrh2", "cdrh3", "cdrh4",
]

