"""
SentencePiece BPE Tokenizer for MewtwoLLM

Trains a BPE tokenizer on collected text data.
32K vocabulary, byte fallback for any Unicode, digit splitting.
"""

import os
import sentencepiece as spm


class MewtwoTokenizer:
    """SentencePiece BPE tokenizer wrapper for MewtwoLLM."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.sp = None
        self._pad_id = 0
        self._unk_id = 1
        self._bos_id = 2
        self._eos_id = 3

    def train(self, input_file: str, model_prefix: str = "mewtwo"):
        """Train a SentencePiece BPE tokenizer."""
        print(f"Training tokenizer with vocab_size={self.vocab_size}...")

        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            byte_fallback=True,
            split_digits=True,
            num_threads=os.cpu_count() or 4,
            pad_id=self._pad_id,
            unk_id=self._unk_id,
            bos_id=self._bos_id,
            eos_id=self._eos_id,
            normalization_rule_name="identity",
        )

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{model_prefix}.model")
        print(f"Tokenizer trained: {model_prefix}.model")
        return self

    @classmethod
    def from_pretrained(cls, path: str = "data/tokenizer"):
        """Load a trained tokenizer from disk."""
        model_path = os.path.join(path, "mewtwo.model") if os.path.isdir(path) else f"{path}.model"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

        tokenizer = cls()
        tokenizer.sp = spm.SentencePieceProcessor()
        tokenizer.sp.load(model_path)
        tokenizer.vocab_size = tokenizer.sp.get_piece_size()
        return tokenizer

    @property
    def pad_id(self):
        return self._pad_id

    @property
    def unk_id(self):
        return self._unk_id

    @property
    def bos_id(self):
        return self._bos_id

    @property
    def eos_id(self):
        return self._eos_id

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True, out_type=None) -> list:
        """Encode text to token IDs.

        Args:
            text: Input text to encode
            add_bos: Whether to add BOS token (currently handled by SentencePiece)
            add_eos: Whether to add EOS token (currently handled by SentencePiece)
            out_type: Output type (for backward compatibility, ignored — always returns int)
        """
        if self.sp is None:
            raise RuntimeError("Tokenizer not trained or loaded. Call train() or from_pretrained().")
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list) -> str:
        """Decode token IDs to text."""
        if self.sp is None:
            raise RuntimeError("Tokenizer not trained or loaded.")
        return self.sp.decode(ids)

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return f"MewtwoTokenizer(vocab_size={self.vocab_size})"


def train_tokenizer(
    input_files: str,
    model_prefix: str = "mewtwo_tokenizer",
    vocab_size: int = 32000,
    character_coverage: float = 0.9995,
):
    """Train a SentencePiece BPE tokenizer (standalone function)."""
    print(f"Training tokenizer with vocab_size={vocab_size}...")

    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        byte_fallback=True,
        split_digits=True,
        num_threads=os.cpu_count() or 4,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        normalization_rule_name="identity",
    )

    print(f"Tokenizer trained: {model_prefix}.model")
    return f"{model_prefix}.model"


def load_tokenizer(model_path: str = "mewtwo_tokenizer.model"):
    """Load a trained SentencePiece tokenizer and return MewtwoTokenizer wrapper."""
    tokenizer = MewtwoTokenizer.__new__(MewtwoTokenizer)
    tokenizer.sp = spm.SentencePieceProcessor()
    tokenizer.sp.load(model_path)
    tokenizer.vocab_size = tokenizer.sp.get_piece_size()
    tokenizer._pad_id = 0
    tokenizer._unk_id = 1
    tokenizer._bos_id = 2
    tokenizer._eos_id = 3
    return tokenizer


def tokenize_file(
    input_path: str,
    output_path: str,
    tokenizer_path: str = "mewtwo_tokenizer.model",
):
    """Tokenize a text file and save as token IDs."""
    tokenizer = load_tokenizer(tokenizer_path)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer.encode(text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, tokens)))

    print(f"Tokenized {input_path} -> {output_path} ({len(tokens)} tokens)")
    return tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MewtwoLLM tokenizer")
    parser.add_argument("--input", required=True, help="Input text file(s)")
    parser.add_argument("--model_prefix", default="mewtwo_tokenizer", help="Output model prefix")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    args = parser.parse_args()

    train_tokenizer(args.input, args.model_prefix, args.vocab_size)
