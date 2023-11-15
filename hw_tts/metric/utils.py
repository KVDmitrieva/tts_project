import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if not target_text:
        return 1 if predicted_text else 0

    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if not target_text:
        return 1 if predicted_text else 0

    target_text_splited = target_text.split(' ')
    predicted_text_splited = predicted_text.split(' ')
    return editdistance.eval(target_text_splited, predicted_text_splited) / len(target_text_splited)