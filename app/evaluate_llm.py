from typing import Dict, Any, List, Tuple
import evaluate
from nltk import edit_distance


def validate_inputs(predictions: str, references: str) -> None:
    """
    Validate that the inputs are strings.

    Args:
        predictions (str): The predicted text.
        references (str): The reference text.

    Raises:
        ValueError: If either argument is not a string.
    """
    if not isinstance(predictions, str) or not isinstance(references, str):
        raise ValueError("Both predictions and references must be strings.")


def parse_and_truncate_lines(predictions: str, references: str) -> Tuple[List[str], List[str]]:
    """
    Split the predictions and references into lines, strip whitespace,
    and truncate both lists to the same length.

    Args:
        predictions (str): The raw predicted text.
        references (str): The raw reference text.

    Returns:
        Tuple[List[str], List[str]]: Truncated lists of prediction and reference lines.
    """
    pred_lines = [line.strip() for line in predictions.splitlines() if line.strip()]
    ref_lines = [line.strip() for line in references.splitlines() if line.strip()]

    min_length = min(len(pred_lines), len(ref_lines))
    return pred_lines[:min_length], ref_lines[:min_length]


def align_line_lengths(
    preds: List[str], refs: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Align the line lengths for consistent edit-distance calculations.

    Args:
        preds (List[str]): List of predicted lines.
        refs (List[str]): List of reference lines.

    Returns:
        Tuple[List[str], List[str]]: Two lists with each line left-justified
        to the maximum length of the lines in both lists.
    """
    max_length = max(
        (len(max(preds, key=len, default="")) if preds else 0),
        (len(max(refs, key=len, default="")) if refs else 0),
    )
    return (
        [seq.ljust(max_length) for seq in preds],
        [seq.ljust(max_length) for seq in refs],
    )


def compute_bleu_score(preds: List[str], refs: List[str]) -> Dict[str, Any]:
    """
    Compute BLEU scores using Hugging Face's evaluate library.

    Args:
        preds (List[str]): List of prediction strings.
        refs (List[str]): List of reference strings.

    Returns:
        Dict[str, Any]: A dictionary containing BLEU score details.
    """
    bleu = evaluate.load("bleu")
    # BLEU expects references as a list of lists
    return bleu.compute(predictions=preds, references=[[ref] for ref in refs])


def compute_rouge_score(preds: List[str], refs: List[str]) -> Dict[str, Any]:
    """
    Compute ROUGE scores using Hugging Face's evaluate library.

    Args:
        preds (List[str]): List of prediction strings.
        refs (List[str]): List of reference strings.

    Returns:
        Dict[str, Any]: A dictionary containing ROUGE score details.
    """
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=preds, references=refs)


def compute_average_med(preds: List[str], refs: List[str]) -> float:
    """
    Compute the average Minimum Edit Distance (MED) for each line pair.

    Args:
        preds (List[str]): List of prediction strings.
        refs (List[str]): List of reference strings.

    Returns:
        float: The average MED score across all line pairs.
    """
    med_scores = [edit_distance(p, r) for p, r in zip(preds, refs)]
    return sum(med_scores) / len(med_scores) if med_scores else 0.0


def compute_metrics(predictions: str, references: str) -> Dict[str, Any]:
    """
    Compute BLEU, ROUGE, and Minimum Edit Distance (MED) scores given predicted text and reference text.

    1. Validates input data.
    2. Splits, strips, and truncates lines.
    3. Optionally aligns line lengths for consistent edit distance.
    4. Computes BLEU, ROUGE, and average MED scores.

    Args:
        predictions (str): Generated sequences (possibly multiple lines).
        references (str): Reference sequences (possibly multiple lines).

    Returns:
        Dict[str, Any]: Dictionary containing:
            - "BLEU": BLEU score details.
            - "ROUGE": ROUGE score details.
            - "MED (Average)": Average minimum edit distance across line pairs.
    """
    validate_inputs(predictions, references)
    pred_lines, ref_lines = parse_and_truncate_lines(predictions, references)
    pred_lines, ref_lines = align_line_lengths(pred_lines, ref_lines)

    bleu_results = compute_bleu_score(pred_lines, ref_lines)
    rouge_results = compute_rouge_score(pred_lines, ref_lines)
    avg_med = compute_average_med(pred_lines, ref_lines)

    return {
        "BLEU": bleu_results,
        "ROUGE": rouge_results,
        "MED (Average)": avg_med,
    }
