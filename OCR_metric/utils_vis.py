from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
from pytesseract import pytesseract
import re
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz
import re
import nltk
from nltk.corpus import words
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error


try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

english_vocab = set(words.words())


tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Adjust this for your system
pytesseract.tesseract_cmd = tesseract_cmd
custom_config = r'--oem 1 --psm 11'  #psm 3,11,6

expected_words = ["Time", "Frequency", "Categories", "Days", "Months", "Years", "Age", "Distance","Value", "Count", "Percentage", "Temperature", "Sales", "Growth", "Intensity", "Score",
                  "Data Analysis", "Monthly Trends", "Distribution Overview", 
                  "Sales Growth", "Temperature Variation", "Frequency Distribution",
                  "Performance Metrics", "Yearly Comparison", "User Activity"]


def is_valid_word(word, min_length=3):
    """Check if word is alphanumeric and long enough."""
    return bool(re.match(r'^[a-zA-Z0-9]+$', word)) and len(word) >= min_length and word.lower() in english_vocab


def get_bounding_boxes(gt_img, min_width=20, min_height=10, vis=True):
    #gt_img = Image.open(img_path)
    #gt_img = preprocess_img(gt_img, binarize=False, grayscale=False,sharpened=True, blurred=True, threshold = 150, vis=False)
    data = pytesseract.image_to_data(gt_img, output_type=pytesseract.Output.DICT,config=custom_config)
    bounding_boxes = []
    detected_texts = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        if len(text) > 0 and w >= min_width and h >= min_height and is_valid_word(text):
            bounding_boxes.append((x, y, w, h))
            detected_texts.append(text)

    if vis:
        gt_image_with_boxes = gt_img.copy()
        draw = ImageDraw.Draw(gt_image_with_boxes)
        for (x, y, w, h) in bounding_boxes:
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(gt_image_with_boxes)
        plt.title(f"Detected {len(bounding_boxes)} bounding boxes after filtering small ones")
        plt.axis("off")
        plt.show()

        print("Sample detected bounding boxes (x, y, w, h):")
        print(bounding_boxes[:5])
        print("Sample detected texts:")
        print(detected_texts[:5])
        
    return bounding_boxes, detected_texts

def get_bounding_boxes2(img_path, min_width=20, min_height=10, vis=True):
    gt_img = Image.open(img_path)
    #gt_img = preprocess_img(gt_img, binarize=False, grayscale=False,sharpened=True, blurred=True, threshold = 150, vis=False)
    data = pytesseract.image_to_data(gt_img, output_type=pytesseract.Output.DICT,config=custom_config)
    bounding_boxes = []
    detected_texts = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        if len(text) > 0 and w >= min_width and h >= min_height and is_valid_word(text):
            bounding_boxes.append((x, y, w, h))
            detected_texts.append(text)

    if vis:
        gt_image_with_boxes = gt_img.copy()
        draw = ImageDraw.Draw(gt_image_with_boxes)
        for (x, y, w, h) in bounding_boxes:
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(gt_image_with_boxes)
        plt.title(f"Detected {len(bounding_boxes)} bounding boxes after filtering small ones")
        plt.axis("off")
        plt.show()

        print("Sample detected bounding boxes (x, y, w, h):")
        print(bounding_boxes[:5])
        print("Sample detected texts:")
        print(detected_texts[:5])
        
    return bounding_boxes, detected_texts

def crop_bounding_boxes(gt_img, sr_img, bounding_boxes, vis=True, num_vis=3):
    cropped_pairs = []
    gt_img = Image.open(gt_img)
    sr_img = Image.open(sr_img)
    for (x, y, w, h) in bounding_boxes:
        gt_crop = gt_img.crop((x, y, x + w, y + h))
        sr_crop = sr_img.crop((x, y, x + w, y + h))
        cropped_pairs.append((gt_crop, sr_crop))

    if vis and cropped_pairs:
        fig, axes = plt.subplots(num_vis, 2, figsize=(8, num_vis * 3))
        for idx in range(min(num_vis, len(cropped_pairs))):
            gt_crop, sr_crop = cropped_pairs[idx]

            axes[idx, 0].imshow(gt_crop, cmap="gray")
            axes[idx, 0].set_title(f"GT Crop {idx+1}")
            axes[idx, 0].axis("off")

            axes[idx, 1].imshow(sr_crop, cmap="gray")
            axes[idx, 1].set_title(f"SR Crop {idx+1}")
            axes[idx, 1].axis("off")

        plt.tight_layout()
        plt.show()
    
    return cropped_pairs

def ocr_2nd_stage(cropped_pairs, vis=True, config = custom_config, num_vis = 3):
    ocr_result = []
    for idx, (gt_crop, sr_crop) in enumerate(cropped_pairs):
        gt_text = pytesseract.image_to_string(gt_crop, config=config)
        sr_text = pytesseract.image_to_string(sr_crop, config=config)
        ocr_result.append({"GT_text": gt_text, "SR_text":sr_text})
    if vis:
        print("Sample OCR Results from Crops:")
        for i in range(min(num_vis, len(ocr_result))):
            print(f"\n--- Crop {i+1} ---")
            print(f"GT Text: {ocr_result[i]['GT_text']}")
            print(f"SR Text: {ocr_result[i]['SR_text']}")

    return ocr_result

def eval_ocr_results(ocr_result, cropped_pairs, verbose=True):
    cer_scores = []
    wer_scores = []
    coverage_scores = []
    soft_coverage_scores = []
    psnr_scores = []
    ssim_scores = []
    mse_scores = []
    
    for i,result in enumerate(ocr_result):
        gt = result['GT_text']
        sr = result['SR_text']
        gt_crop, sr_crop = cropped_pairs[i]
        
        cer = calc_CER(gt, sr)
        wer = calc_WER(gt, sr)
        coverage = calc_text_coverage(gt, sr)
        soft_coverage = calc_soft_coverage(gt, sr)
        
        psnr_val, ssim_val, mse_val = compute_pixel_metrics(gt_crop, sr_crop)
        
        cer_scores.append(cer)
        wer_scores.append(wer)
        coverage_scores.append(coverage)
        soft_coverage_scores.append(soft_coverage)
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)
        mse_scores.append(mse_val)
        
        if verbose and i < 3:  
            print(f"\n--- Crop {i+1} ---")
            print(f"GT: {gt}")
            print(f"SR: {sr}")
            print(f"CER: {cer:.3f}, WER: {wer:.3f}, Coverage: {coverage:.3f}, Soft Coverage: {soft_coverage:.3f}")
            print(f"PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.3f}, MSE: {mse_val:.2f}")


    return {
        "Mean CER": sum(cer_scores) / len(cer_scores),
        "Mean WER": sum(wer_scores) / len(wer_scores),
        "Mean Coverage": sum(coverage_scores) / len(coverage_scores),
        "Mean Soft Coverage": sum(soft_coverage_scores) / len(soft_coverage_scores),
        "Mean PSNR": np.mean(psnr_scores),
        "Mean SSIM": np.mean(ssim_scores),
        "Mean MSE": np.mean(mse_scores),
    }

def compute_pixel_metrics(gt_crop, sr_crop):
    gt_np = np.array(gt_crop.convert("L"))
    sr_np = np.array(sr_crop.convert("L"))

    mse_val = mean_squared_error(gt_np.flatten(), sr_np.flatten())
    psnr_val = psnr(gt_np, sr_np, data_range=255)
    ssim_val = ssim(gt_np, sr_np, data_range=255)

    return psnr_val, ssim_val, mse_val
    

def calc_score_v2(gt_path, sr_path, vis=False):
    gt_img = Image.open(gt_path)
    sr_img = Image.open(sr_path)
    gt_img = preprocess_img(gt_img, binarize=False, grayscale=True, sharpened=True, blurred=True, threshold=150, vis=vis)
    sr_img = preprocess_img(sr_img, binarize=False, grayscale=True, sharpened=True, blurred=True, threshold=150, vis=vis)
    bounding_boxes, detected_texts = get_bounding_boxes(gt_img,vis=vis)
    cropped_bounding_boxes = crop_bounding_boxes(gt_img,sr_img, bounding_boxes, vis=vis)
    ocr_res = ocr_2nd_stage(cropped_bounding_boxes, vis=vis)
    metrics = eval_ocr_results(ocr_res, cropped_bounding_boxes)
    print("\n=== OCR Evaluation Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")

        

def preprocess_img(img, binarize=False, grayscale=False,sharpened=False, blurred=False, threshold = 150, vis=False):
    applied_steps = []
    if grayscale or binarize:
        img = img.convert('L')
        applied_steps.append("Grayscale")
    if binarize:
        img = img.point(lambda p: 255 if p > threshold else 0)
        applied_steps.append("Binarized")
    if sharpened:
        img = img.filter(ImageFilter.SHARPEN)
        applied_steps.append("Sharpened")
    if blurred:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        applied_steps.append("Blurred")
    if vis:
        plt.figure(figsize=(5, 5))
        plt.imshow(img,cmap='gray')
        title = " | ".join(applied_steps) if applied_steps else "Original Image"
        plt.title(title)
        plt.axis("off")
        plt.show()
    return img

def extract_text(img_path=None, img=None,print=False):
    if img == None:
        image = Image.open(img_path)
    else:
        image = img
    text = pytesseract.image_to_string(image)
    if print:
        print("Extracted Text:\n", text)
    return text
    
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = text.split()
    filtered_words = [word for word in words if len(word) > 2 or word in expected_words]
    final_text = " ".join(word for word in filtered_words if word in expected_words)
    return final_text

def extraction_pipeline(img_path=None, img=None, binarize=False, grayscale=False,sharpened=False, blurred=False, threshold = 150, vis=False):
    img = preprocess_img(img, binarize, grayscale,sharpened, blurred, threshold, vis)
    text = extract_text(img_path, img)
    cleaned_text = clean_text(text)
    return cleaned_text

def calc_WER(gt_text, ocr_text):
    if not gt_text or not ocr_text:
        wer = 1.0
    else:
        gt_words = gt_text.split() #needs rework, should be used instead of text
        ocr_words = ocr_text.split()
        wer = Levenshtein.distance(gt_words, ocr_words) / max(len(gt_words), len(ocr_words), 1)
    return wer

def calc_CER(gt_text, ocr_text):
    if not gt_text or not ocr_text:
        cer = 1.0
    else:
        cer = Levenshtein.distance(gt_text, ocr_text) / max(len(gt_text), len(ocr_text),1)
    return cer

def calc_text_coverage(gt_text, ocr_text, threshold=60):
    if not gt_text or not ocr_text:
        coverage = 0.0
    else:
        gt_words = set(re.sub(r'\W+', '', word.lower()) for word in gt_text.split())
        ocr_words = set(re.sub(r'\W+', '', word.lower()) for word in ocr_text.split())
        
        match_count = 0
    
        for ocr_word in ocr_words:
            scores = [fuzz.ratio(ocr_word, gt_word) for gt_word in gt_words]
            #print(f"{ocr_word}: scores = {scores}")
            best_score = max([fuzz.ratio(ocr_word, gt_word) for gt_word in gt_words] or [0])
            if best_score >= threshold:
                match_count += 1
        coverage = match_count / max(len(gt_words), 1)
    return coverage

def calc_soft_coverage(gt_text, ocr_text):
    if not gt_text or not ocr_text:
        return 0.0

    gt_words = set(gt_text.lower().split())
    ocr_words = set(ocr_text.lower().split())

    fuzzy_total = 0.0

    for ocr_word in ocr_words:
        best_score = max([fuzz.ratio(ocr_word, gt_word) for gt_word in gt_words] or [0])
        fuzzy_total += best_score / 100  # Normalize

    return fuzzy_total / max(len(gt_words), 1)

